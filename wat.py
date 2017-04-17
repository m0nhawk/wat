import os
import re

import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.widgets.MatplotlibWidget as mpw
import scipy
import wavelets
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QGridLayout, QAbstractItemView, QPushButton,
                             QDialog, QVBoxLayout)

import ui_main
from models import PandasModel
from physics import cyclotron_frequency


class PlotDataWindow(QDialog):
    def __init__(self, data, time_axis, data_axis, date_format='', parent=None):
        super(PlotDataWindow, self).__init__(parent)

        self.plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 2.0))
        self.layout = QGridLayout()
        self.layout.addWidget(self.plot)
        self.setLayout(self.layout)

        self.plot_data(data, time_axis, data_axis, date_format)

    def plot_data(self, data, time_axis, data_axis, date_format=''):
        time_slice = pd.to_datetime(data[time_axis], unit='s')
        if date_format == '':
            self.time = pd.to_datetime(data[time_axis], unit='s')
        else:
            self.time = pd.to_datetime(data[time_axis], format=date_format)
        b_slice = data[data_axis].values

        ax = self.plot.getFigure().add_subplot(111)
        ax.plot(time_slice, b_slice)

        ax.set_xlabel('time, UT')
        ax.set_xlim([time_slice.iloc[0], time_slice.iloc[-1]])
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=960))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.set_ylabel(data_axis)
        ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
        ax.grid(True)

        self.plot.getFigure().set_tight_layout(True)


class WaveletPlotWindow(QDialog):
    def __init__(self, data, time_axis, data_axis, elements=None, date_format='', parent=None):
        super(WaveletPlotWindow, self).__init__(parent)

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")

        self.save_cyclotron_button = QPushButton(self)
        self.save_cyclotron_button.setText("Save Cyclotron")
        self.verticalLayout.addWidget(self.save_cyclotron_button)
        self.save_cyclotron_button.clicked.connect(self.save_cyclotron)

        self.plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 9.0))
        self.verticalLayout.addWidget(self.plot)

        self.time = None
        self.cyclotron_power = {}
        self.cyclotron_period = {}

        self.open_folder_path = os.path.expanduser(r'~\Documents\me\science\volkswagen_grant\jupiter\converted')

        self.plot_wavelet(data, time_axis, data_axis, elements=elements, date_format=date_format)

    def save_cyclotron(self):
        filename, ok = QFileDialog.getSaveFileName(self, 'Save file',
                                                   self.open_folder_path,
                                                   "Text files (*.txt *.asc *.dat *.csv)")

        if ok:
            res = {'time': self.time}

            for element in self.elements:
                res[element['name']] = self.cyclotron_power[element['name']]

            result = pd.DataFrame(res)
            result.to_csv(filename, index=False, na_rep=0)

    def plot_wavelet(self, data, time_axis, data_axis, title='', elements=None, date_format=''):
        xlabel = data_axis

        magnetic_field = data[data_axis].values

        if date_format == '':
            self.time = pd.to_datetime(data[time_axis], unit='s')
        else:
            self.time = pd.to_datetime(data[time_axis], format=date_format)

        wavelet_time = self.time.apply(lambda x: x.tz_localize('utc').timestamp()).values

        dt = wavelet_time[1] - wavelet_time[0]
        wa = wavelets.WaveletAnalysis(data=magnetic_field, time=wavelet_time, dt=dt, dj=0.125,
                                      wavelet=wavelets.Morlet(), unbias=True)
        power = wa.wavelet_power
        t, scales = wa.time, wa.scales

        grid = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 4])

        fig = self.plot.getFigure()

        def date_formatter(x, pos):
            return pd.to_datetime(x, unit='s').strftime('%H:%M')

        formatter = ticker.FuncFormatter(date_formatter)

        ax_magnetic = self.plot.getFigure().add_subplot(grid[0])
        ax_magnetic.plot(t, magnetic_field)

        ax_magnetic.set_title(title)

        ax_magnetic.set_xlabel('time, UT')
        ax_magnetic.set_xlim([t[0], t[-1]])
        ax_magnetic.xaxis.set_major_locator(ticker.AutoLocator())
        ax_magnetic.xaxis.set_major_formatter(formatter)

        ax_magnetic.set_ylabel(xlabel)
        ax_magnetic.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

        ax_magnetic.grid(True)

        vmin = power.min()
        vmax = power.max()

        locator = ticker.LinearLocator(numticks=64)
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

        C, S = wa.coi
        C = np.insert(C, [0, C.size], [t.min(), t.max()])
        S = np.insert(S, [0, S.size], [0, 0])

        scales = np.array([s for s in scales if s < S.max()])
        power = power[0:len(scales)]

        interpolated_coi = scipy.interpolate.interp1d(C, S, bounds_error=False)

        def find_nearest_idx(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        for element in elements:
            charge = element['charge']
            mass = element['mass']
            name = element['name']

            cyclotron_period = 1 / cyclotron_frequency(magnetic_field, charge, mass)
            cyclotron_period[cyclotron_period > interpolated_coi(t)] = np.nan

            self.cyclotron_period[name] = cyclotron_period

            tt = np.arange(0.0, len(t))
            ss = np.array([find_nearest_idx(scales, x) for x in cyclotron_period], dtype=float)

            ss[ss == 0] = np.nan
            cyclotron_power = scipy.ndimage.map_coordinates(power, np.vstack((ss, tt)), order=0)
            cyclotron_power[cyclotron_power == 0] = np.nan

            self.cyclotron_power[name] = cyclotron_power

        ax_cyclotron = self.plot.getFigure().add_subplot(grid[1])
        ax_cyclotron.semilogy(t, self.cyclotron_power[elements[0]['name']])

        ax_cyclotron.set_title('Power on cyclotron frequency')

        ax_cyclotron.set_xlabel('time, UT')
        ax_cyclotron.set_xlim([t[0], t[-1]])
        ax_cyclotron.xaxis.set_major_locator(ticker.AutoLocator())
        ax_cyclotron.xaxis.set_major_formatter(formatter)

        ax_cyclotron.set_ylabel('Power, (nT)^2')
        ax_cyclotron.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

        ax_cyclotron.grid(True)

        T, S = np.meshgrid(t, scales)

        ax_wavelet = self.plot.getFigure().add_subplot(grid[2])

        s = ax_wavelet.contourf(T, S, power, np.arange(vmin, vmax, 0.001), locator=locator, norm=norm)

        cb = fig.colorbar(s, ax=ax_wavelet, orientation='horizontal', pad=0.2, extend='both')
        cb.set_label('Wavelet power spectrum, (nT)^2')
        cb.set_clim(vmin=vmin, vmax=vmax)

        ax_wavelet.set_xlabel('time, UT')
        ax_wavelet.set_xlim([t[0], t[-1]])
        ax_wavelet.xaxis.set_major_locator(ticker.AutoLocator())
        ax_wavelet.xaxis.set_major_formatter(formatter)

        C, S = wa.coi
        C = np.insert(C, [0, C.size], [t.min(), t.max()])
        S = np.insert(S, [0, S.size], [0, 0])

        ax_wavelet.set_ylabel('scale, s')
        ax_wavelet.set_ylim(scales.max(), scales.min())
        ax_wavelet.set_yscale('log')
        ax_wavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())

        ax_wavelet.fill_between(x=C, y1=S, y2=scales.max(), color='gray', alpha=0.3)
        ax_wavelet.plot(t, self.cyclotron_period[elements[0]['name']], 'r-', linewidth=1)

        ax_wavelet.grid(b=True, which='major', color='k', linestyle='-', alpha=.2, zorder=3)

        ax_wavelet_fourier = ax_wavelet.twinx()

        ax_wavelet_fourier.set_ylabel('frequency, Hz')
        fourier_lim = [1 / wa.fourier_period(i) for i in ax_wavelet.get_ylim()]
        ax_wavelet_fourier.set_ylim(fourier_lim)
        ax_wavelet_fourier.set_yscale('log')
        ax_wavelet_fourier.set_yticks([10 ** -1, 10 ** -2, 5 * 10 ** -3, 10 ** -3])

        self.plot.getFigure().set_tight_layout(True)


class WaveletAnalysisApp(QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self):
        super(WaveletAnalysisApp, self).__init__()
        self.setupUi(self)

        self.buttonOpenDataFile.clicked.connect(self.openDataFile)
        self.buttonPlotData.clicked.connect(self.plotData)
        self.buttonPlotWavelet.clicked.connect(self.plotWavelet)

        self.data = None
        self.disabled = True

        self.dataModel = PandasModel()
        self.dataView.setModel(self.dataModel)

        self.dataView.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.dataView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.dataView.verticalHeader().hide()

        self.open_folder_path = os.path.expanduser(r'~\Documents\me\science\volkswagen_grant\jupiter\converted')

    def openDataFile(self):
        filename, ok = QFileDialog.getOpenFileName(None, 'Open file',
                                                   self.open_folder_path,
                                                   "Text files (*.txt *.asc *.dat *.csv)")

        self.open_folder_path = os.path.dirname(filename)
        self.open_file_name = os.path.basename(filename)

        if ok:
            self.data = pd.read_csv(filename, sep=None, engine='python')
            self.data.fillna(0, inplace=True)

            self.dataModel = PandasModel(self.data)
            self.dataView.setModel(self.dataModel)

            if self.disabled:
                disabled_items = [self.listTime, self.listData, self.buttonPlotData, self.buttonPlotWavelet,
                                  self.dataView, self.chemical]
                for item in disabled_items:
                    item.setEnabled(True)
                self.disabled = False

            self.listTime.clear()
            self.listData.clear()
            self.listTime.addItems(self.data.columns)
            self.listData.addItems(self.data.columns)

            self.listData.setCurrentIndex(1)

            self.currentFile.setText(self.open_file_name)

    def plotData(self):
        plt = PlotDataWindow(self.data, self.listTime.currentText(), self.listData.currentText(),
                             self.dateFormat.text(), parent=self)
        plt.show()

    def plotWavelet(self):
        element_string = self.chemical.text()

        elements_list = element_string.split(',')

        elements = []

        for element in elements_list:
            charge = element.count("+")

            element_sym = re.findall('([A-Z][a-z]?)+', element)[0]

            elements_mass = {'H': 1, 'He': 4, 'Li': 7, 'Be': 9, 'B': 11, 'C': 12, 'N': 14, 'O': 16, 'F': 19, 'Ne': 20,
                             'Na': 23, 'Mg': 24, 'Al': 27, 'Si': 28, 'P': 30, 'S': 32, 'Cl': 35.5, 'Ar': 40, 'K': 40,
                             'Ca': 40}

            mass = elements_mass[element_sym]

            elements.append({'name': element, 'mass': mass, 'charge': charge})

        plt = WaveletPlotWindow(self.data, self.listTime.currentText(), self.listData.currentText(), elements=elements,
                                date_format=self.dateFormat.text(), parent=self)
        plt.show()
