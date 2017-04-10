import datetime
import os
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
from PyQt5.QtCore import QRect, QPoint, QItemSelectionModel, QEvent
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog, QWidget, QGridLayout, QAbstractItemView)

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


class WaveletPlotWindow(QMainWindow):
    def __init__(self, data, time_axis, data_axis, element=1, parent=None):
        super(WaveletPlotWindow, self).__init__(parent)

        w = QWidget()

        layout = QGridLayout()
        w.setLayout(layout)

        plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 9.0))

        layout.addWidget(plot)

        title = ''
        xlabel = ''
        log = False

        time_slice = pd.to_datetime(data[time_axis], unit='s')
        b_slice = data[data_axis].values

        x = b_slice
        dt = time_slice.iloc[1] - time_slice.iloc[0]
        dt = float('.'.join([str(dt.seconds), str(dt.microseconds)]))
        wa = wavelets.WaveletAnalysis(x, dt=dt, dj=0.125, wavelet=wavelets.Morlet(), unbias=True)
        power = wa.wavelet_power
        scales = wa.scales
        t = wa.time

        grid = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 4])

        fig = plot.getFigure()

        ax1 = plot.getFigure().add_subplot(grid[0])

        ax1.set_title(title)
        ax1.plot(time_slice, b_slice)
        ax1.xaxis.set_major_locator(mdates.SecondLocator(interval=960))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.set_xlim([time_slice.iloc[0], time_slice.iloc[-1]])
        ax1.set_xlabel('time, UT')
        ax1.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
        ax1.set_ylabel(xlabel)
        ax1.grid(True)

        C, S = wa.coi
        scales = np.array([s for s in scales if s < S.max()])

        power = power[0:len(scales)]
        T, S = np.meshgrid(t, scales)

        vmin = power.min()
        vmax = power.max()

        if log:
            locator = ticker.LogLocator(numticks=64)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
        else:
            locator = ticker.LinearLocator(numticks=64)
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

        Tt, St = wa.coi
        interpolated_coi = scipy.interpolate.interp1d(Tt, St, bounds_error=False)

        cyclotron_period = 1 / cyclotron_frequency(x, element)
        cyclotron_period[cyclotron_period > interpolated_coi(t)] = np.nan

        def find_nearest_idx(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        tt = np.arange(0.0, len(t))
        ss = np.array([find_nearest_idx(scales, x) for x in cyclotron_period], dtype=float)

        ss[ss == 0] = np.nan
        cyclotron_power = scipy.ndimage.map_coordinates(power, np.vstack((ss, tt)), order=0)
        cyclotron_power[cyclotron_power == 0] = np.nan

        ax0 = plot.getFigure().add_subplot(grid[1])

        ax0.set_title('Power on cyclotron frequency')
        ax0.semilogy(time_slice, cyclotron_power)

        ax0.xaxis.set_major_locator(mdates.SecondLocator(interval=960))
        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax0.set_xlim([time_slice.iloc[0], time_slice.iloc[-1]])
        ax0.set_xlabel('time, UT')
        ax0.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
        ax0.set_ylabel('Power, (nT)^2')
        ax0.grid(True)

        ax2 = plot.getFigure().add_subplot(grid[2])

        s = ax2.contourf(T, S, power, np.arange(vmin, vmax, 0.001), locator=locator, norm=norm, vmin=vmin, vmax=vmax)
        ax2.set_xlabel('time, UT')

        def wavelet_date_formatter(x, pos):
            return (time_slice[0] + datetime.timedelta(0, float(x))).strftime('%H:%M')

        formatter = ticker.FuncFormatter(wavelet_date_formatter)
        ax2.get_xaxis().set_major_formatter(formatter)
        ax2.grid(b=True, which='major', color='k', linestyle='--', alpha=.5, zorder=3)

        C, S = wa.coi

        ax2.set_ylabel('scale, s')
        ax2.set_yscale('log')
        ax2.set_ylim(scales.max(), scales.min())
        ax2.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax2.fill_between(x=C, y1=S, y2=scales.max(), color='gray', alpha=0.3)

        ax2.plot(t, cyclotron_period, 'r-', linewidth=1)

        cb = fig.colorbar(s, ax=ax2, orientation='horizontal', pad=0.2, extend='both')
        cb.set_label('Wavelet power spectrum, (nT)^2')
        cb.set_clim(vmin=vmin, vmax=vmax)

        ax_fourier = ax2.twinx()

        ax_fourier.set_yscale('log')
        ax_fourier.set_yticks([10 ** -1, 10 ** -2, 5 * 10 ** -3, 10 ** -3])
        ax_fourier.set_ylabel('frequency, Hz')

        fourier_lim = [1 / wa.fourier_period(i) for i in ax2.get_ylim()]
        ax_fourier.set_ylim(fourier_lim)

        plot.getFigure().set_tight_layout(True)

        self.setCentralWidget(w)


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
            bgsm_c1 = pd.read_csv(filename, sep=None, engine='python')
            # bgsm_c1.ix[:, 0] = pd.to_datetime(bgsm_c1.ix[:, 0], infer_datetime_format=True).apply(lambda x: x.timestamp())

            self.data = bgsm_c1
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
        element = self.chemical.currentIndex()

        if element == 0:
            element = 1
        elif element == 1:
            element = 4
        elif element == 2:
            element = 16
        else:
            element = 1

        plt = WaveletPlotWindow(self.data, self.time_axis, self.data_axis, element=element, parent=self)
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
