import os
import re

import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.widgets.MatplotlibWidget as mpw
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QGridLayout, QAbstractItemView, QPushButton,
                             QDialog, QVBoxLayout, QLineEdit)
from pandas.plotting import _converter

import ui_main
from models import PandasModel
from physics import wavelet

_converter.register()


class PlotDataWindow(QDialog):
    def __init__(self, parent=None):
        super(PlotDataWindow, self).__init__(parent)

        self._plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 2.0))
        self.layout = QGridLayout()
        self.layout.addWidget(self._plot)
        self.setLayout(self.layout)

    def plot(self, time, field, fig=None, sign=None):
        ax = fig.add_subplot(111)
        ax.plot(time, field)

        ax.set_xlabel(sign['xlabel'])
        ax.set_xlim(time[0], time[-1])
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=960))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.set_ylabel(sign['ylabel'])
        ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

        ax.grid(True)

        fig.set_tight_layout(True)

    def plot_data(self, data, time_axis, data_axis, date_format=''):
        if date_format:
            time = pd.to_datetime(data[time_axis], format=date_format)
        else:
            time = pd.to_datetime(data[time_axis], unit='s')

        time = time.values
        field = data[data_axis].values

        fig = self._plot.getFigure()

        self.plot(time, field, fig=fig, sign={'xlabel': 'time, UT', 'ylabel': data_axis})


class WaveletPlotWindow(QDialog):
    def __init__(self, data, time_axis, data_axis, elements=None, date_format='', parent=None):
        super(WaveletPlotWindow, self).__init__(parent)

        self.resize(900, 900)

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")

        self.save_cyclotron_button = QPushButton(self)
        self.save_cyclotron_button.setText("Save Cyclotron")
        self.verticalLayout.addWidget(self.save_cyclotron_button)
        self.save_cyclotron_button.clicked.connect(self.save_cyclotron)

        self.integral_value = QLineEdit(self)
        self.verticalLayout.addWidget(self.integral_value)

        self.plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 9.0))
        self.verticalLayout.addWidget(self.plot)

        self.time = None
        self.cyclotron_power = {}
        self.elements = elements

        self.open_folder_path = os.path.expanduser(r'~\Documents\me\science\volkswagen_grant\wavelet_cd')

        self.plot_wavelet(data, time_axis, data_axis, elements=elements, date_format=date_format,
                          title='Wavelet modulus')

    def save_cyclotron(self):
        filename, ok = QFileDialog.getSaveFileName(self, 'Save file',
                                                   self.open_folder_path,
                                                   "Text files (*.txt *.asc *.dat *.csv)")

        if ok:
            res = {'time': self.time}

            if self.elements:
                for element in self.elements:
                    res[element['name']] = self.cyclotron_power[element['name']]

            result = pd.DataFrame(res)
            result.to_csv(filename, index=False, na_rep=0)

    def plot_wavelet(self, data, time_axis, data_axis, title='', elements=None, date_format=''):
        magnetic_field = data[data_axis].values

        if date_format == '':
            self.time = pd.to_datetime(data[time_axis], unit='s')
        else:
            self.time = pd.to_datetime(data[time_axis], format=date_format)

        wavelet_time = self.time.apply(lambda x: x.tz_localize('utc').timestamp()).values

        fig = self.plot.getFigure()

        self.cyclotron_power, self.integral = wavelet(wavelet_time, magnetic_field, elements=elements, fig=fig,
                                                      sign={'title': title, 'xlabel': 'time (H:M:S)',
                                                            'ylabel': 'Field (nT)'})

        self.integral_value.setText(str(self.integral))


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

        self.open_folder_path = os.path.expanduser(r'~\Documents\me\science\volkswagen_grant\B phi')

    def openDataFile(self):
        filename, ok = QFileDialog.getOpenFileName(None, 'Open file',
                                                   self.open_folder_path,
                                                   filter="Text files (*.txt *.asc *.dat *.csv *.cef);;All files (*.*)")

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
        range = self.range.text()
        if range:
            begin = range.split(":")[0]
            end = range.split(":")[1]
        else:
            begin = 0
            end = self.data.shape[0]

        plt = PlotDataWindow(parent=self)
        plt.plot_data(self.data.loc[begin:end], self.listTime.currentText(), self.listData.currentText(),
                      self.dateFormat.currentText())
        plt.show()

    def plotWavelet(self):
        range = self.range.text()
        if range:
            begin = range.split(":")[0]
            end = range.split(":")[1]
        else:
            begin = 0
            end = self.data.shape[0]

        elements = self.chemical.text()
        elements = elements.split(',')

        elements_dict = []

        for element in elements:
            if not element:
                continue
            charge = element.count("+")

            element_sym = re.findall('([A-Z][a-z]?)+', element)[0]

            elements_mass = {'H': 1, 'He': 4, 'Li': 7, 'Be': 9, 'B': 11, 'C': 12, 'N': 14, 'O': 16, 'F': 19, 'Ne': 20,
                             'Na': 23, 'Mg': 24, 'Al': 27, 'Si': 28, 'P': 30, 'S': 32, 'Cl': 35.5, 'Ar': 40, 'K': 40,
                             'Ca': 40}

            mass = elements_mass[element_sym]

            elements_dict.append({'name': element, 'mass': mass, 'charge': charge})

        plt = WaveletPlotWindow(self.data.loc[begin:end], self.listTime.currentText(), self.listData.currentText(),
                                elements=elements_dict, date_format=self.dateFormat.currentText(), parent=self)
        plt.show()
