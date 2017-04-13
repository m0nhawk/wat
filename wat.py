import datetime
import os
import re
import sys

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
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog, QWidget, QGridLayout, QAbstractItemView, QPushButton, QDialog, QVBoxLayout, QSizePolicy)

import ui_main
from models import PandasModel
from physics import cyclotron_frequency


class DataPlotWindow(QMainWindow):
    def __init__(self, data, time_axis, data_axis, parent=None):
        super(DataPlotWindow, self).__init__(parent)

        w = QWidget()

        layout = QGridLayout()
        w.setLayout(layout)

        plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 9.0))

        layout.addWidget(plot)

        time_slice = pd.to_datetime(data[time_axis], unit='s')
        b_slice = data[data_axis].values

        ax1 = plot.getFigure().add_subplot(111)

        ax1.plot(time_slice, b_slice)
        ax1.xaxis.set_major_locator(mdates.SecondLocator(interval=960))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.set_xlim([time_slice.iloc[0], time_slice.iloc[-1]])
        ax1.set_xlabel('time, UT')
        ax1.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

        plot.draw()

        self.setCentralWidget(w)


class WaveletPlotWindow(QDialog):
    def __init__(self, data, time_axis, data_axis, elements=None, date_format='', parent=None):
        super(WaveletPlotWindow, self).__init__(parent)

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QPushButton(self)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Save Cyclotron")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton.clicked.connect(self.saveCyclotron)

        self.plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 9.0))
        self.verticalLayout.addWidget(self.plot)

        self.data = data
        self.time_axis = time_axis
        self.data_axis = data_axis
        self.elements = elements
        self.date_format = date_format

        self.cyclotron_power = {}
        self.cyclotron_period = {}
        self.time = None

        self.open_folder_path = os.path.expanduser(r'~\Documents\me\science\volkswagen_grant\jupiter\converted')

        self.wavelet()

    def saveCyclotron(self):
        filename, ok = QFileDialog.getSaveFileName(self, 'Save file',
                                                   self.open_folder_path,
                                                   "Text files (*.txt *.asc *.dat *.csv)")

        if ok:
            res = {'time': self.time.apply(lambda t: t.timestamp())}

            for element in self.elements:
                res[element['name']] = self.cyclotron_power[element['name']]

            result = pd.DataFrame(res)
            result.to_csv(filename, index=False, na_rep=0)

    def wavelet(self):
        title = ''
        xlabel = ''
        log = False

        if self.date_format == '':
            self.time = pd.to_datetime(self.data[self.time_axis], unit='s')
        else:
            self.time = pd.to_datetime(self.data[self.time_axis], format=self.dateFormat.text()).apply(lambda x: x.timestamp())

        magnetic_field = self.data[self.data_axis].values

        dt = (self.time.iloc[1] - self.time.iloc[0]).total_seconds()
        wa = wavelets.WaveletAnalysis(magnetic_field, dt=dt, dj=0.125, wavelet=wavelets.Morlet(), unbias=True)
        power = wa.wavelet_power
        scales = wa.scales
        t = wa.time

        grid = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 4])

        fig = self.plot.getFigure()

        ax_magnetic = self.plot.getFigure().add_subplot(grid[0])

        ax_magnetic.set_title(title)
        ax_magnetic.plot(self.time, magnetic_field)
        ax_magnetic.xaxis.set_major_locator(mdates.SecondLocator(interval=960))
        ax_magnetic.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_magnetic.set_xlim([self.time.iloc[0], self.time.iloc[-1]])
        ax_magnetic.set_xlabel('self.time, UT')
        ax_magnetic.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
        ax_magnetic.set_ylabel(xlabel)
        ax_magnetic.grid(True)

        C, S = wa.coi
        scales = np.array([s for s in scales if s < S.max()])

        vmin = power.min()
        vmax = power.max()

        if log:
            locator = ticker.LogLocator(numticks=64)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
        else:
            locator = ticker.LinearLocator(numticks=64)
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

        C, S = wa.coi
        interpolated_coi = scipy.interpolate.interp1d(C, S, bounds_error=False)

        def find_nearest_idx(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        for element in self.elements:
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

        power = power[0:len(scales)]
        T, S = np.meshgrid(t, scales)

        ax_cyclotron = self.plot.getFigure().add_subplot(grid[1])

        ax_cyclotron.set_title('Power on cyclotron frequency')
        ax_cyclotron.semilogy(self.time, self.cyclotron_power[self.elements[0]['name']])

        ax_cyclotron.xaxis.set_major_locator(mdates.SecondLocator(interval=960))
        ax_cyclotron.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_cyclotron.set_xlim([self.time.iloc[0], self.time.iloc[-1]])
        ax_cyclotron.set_xlabel('self.time, UT')
        ax_cyclotron.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
        ax_cyclotron.set_ylabel('Power, (nT)^2')
        ax_cyclotron.grid(True)

        ax_wavelet = self.plot.getFigure().add_subplot(grid[2])

        s = ax_wavelet.contourf(T, S, power, np.arange(vmin, vmax, 0.001), locator=locator, norm=norm, vmin=vmin,
                                vmax=vmax)
        ax_wavelet.set_xlabel('time, UT')

        def wavelet_date_formatter(x, pos):
            return (self.time[0] + datetime.timedelta(0, float(x))).strftime('%H:%M')

        formatter = ticker.FuncFormatter(wavelet_date_formatter)
        ax_wavelet.get_xaxis().set_major_formatter(formatter)
        ax_wavelet.grid(b=True, which='major', color='k', linestyle='--', alpha=.5, zorder=3)

        C, S = wa.coi

        ax_wavelet.set_ylabel('scale, s')
        ax_wavelet.set_yscale('log')
        ax_wavelet.set_ylim(scales.max(), scales.min())
        ax_wavelet.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax_wavelet.fill_between(x=C, y1=S, y2=scales.max(), color='gray', alpha=0.3)

        ax_wavelet.plot(t, self.cyclotron_period[self.elements[0]['name']], 'r-', linewidth=1)

        cb = fig.colorbar(s, ax=ax_wavelet, orientation='horizontal', pad=0.2, extend='both')
        cb.set_label('Wavelet power spectrum, (nT)^2')
        cb.set_clim(vmin=vmin, vmax=vmax)

        ax_wavelet_fourier = ax_wavelet.twinx()

        ax_wavelet_fourier.set_yscale('log')
        ax_wavelet_fourier.set_yticks([10 ** -1, 10 ** -2, 5 * 10 ** -3, 10 ** -3])
        ax_wavelet_fourier.set_ylabel('frequency, Hz')

        fourier_lim = [1 / wa.fourier_period(i) for i in ax_wavelet.get_ylim()]
        ax_wavelet_fourier.set_ylim(fourier_lim)

        self.plot.getFigure().set_tight_layout(True)


class WaveletAnalysisApp(QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self):
        super(WaveletAnalysisApp, self).__init__()
        self.setupUi(self)

        self.buttonOpenDataFile.clicked.connect(self.openDataFile)
        self.listTime.currentIndexChanged.connect(self.listTimeChanged)
        self.listData.currentIndexChanged.connect(self.listDataChanged)
        self.buttonPlotData.clicked.connect(self.plotData)
        self.buttonPlotWavelet.clicked.connect(self.plotWavelet)

        self.data = None

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

            if not self.listTime.isEnabled():
                self.listTime.setEnabled(True)

            if not self.listData.isEnabled():
                self.listData.setEnabled(True)

            if not self.buttonPlotData.isEnabled():
                self.buttonPlotData.setEnabled(True)

            if not self.buttonPlotWavelet.isEnabled():
                self.buttonPlotWavelet.setEnabled(True)

            if not self.dataView.isEnabled():
                self.dataView.setEnabled(True)

            if not self.chemical.isEnabled():
                self.chemical.setEnabled(True)

            self.listTime.clear()
            self.listData.clear()
            self.listTime.addItems(self.data.columns)
            self.listData.addItems(self.data.columns)

            self.listData.setCurrentIndex(1)

            self.currentFile.setText(self.open_file_name)

    def plotData(self):
        plt = DataPlotWindow(self.data, self.time_axis, self.data_axis, parent=self)
        plt.show()

    def plotWavelet(self):
        element_string = self.chemical.text()

        elements_list = element_string.split(',')

        elements = []

        for element in elements_list:
            charge = element.count("+")

            element_sym = re.findall('([A-Z][a-z]?)+', element)[0]

            elements_mass = {'H': 1, 'He': 4, 'Li': 7, 'Be': 9, 'B': 11, 'C': 12, 'N': 14, 'O': 16, 'F': 19, 'Ne': 20, 'Na': 23,
                        'Mg': 24, 'Al': 27, 'Si': 28, 'P': 30, 'S': 32, 'Cl': 35.5, 'Ar': 40, 'K': 40, 'Ca': 40}

            mass = elements_mass[element_sym]

            elements.append({'name': element, 'mass': mass, 'charge': charge})

        plt = WaveletPlotWindow(self.data, self.time_axis, self.data_axis, elements=elements,
                                date_format=self.dateFormat.text(), parent=self)
        plt.show()

    def listTimeChanged(self, i):
        self.time_axis = self.listTime.currentText()

    def listDataChanged(self, i):
        self.data_axis = self.listData.currentText()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = WaveletAnalysisApp()
    form.show()
    app.exec_()
