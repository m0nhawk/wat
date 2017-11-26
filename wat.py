import gettext
import os
import re

import matplotlib.dates as dates
import matplotlib.ticker as ticker
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.widgets.MatplotlibWidget as mpw
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QGridLayout, QAbstractItemView, QPushButton,
                             QDialog, QVBoxLayout, QLineEdit)
from pandas.plotting import _converter

import helpers
import ui_main
from models import PandasModel
from physics import wavelet

localedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'locale')
translate = gettext.translation('wat', localedir, fallback=True)
_ = translate.gettext

_converter.register()


class FieldPlotWindow(QDialog):
    def __init__(self, time, field, date_format=None, labels=None, parent=None):
        super(FieldPlotWindow, self).__init__(parent)

        self.time = helpers.get_formatted_time(time, date_format=date_format).values
        self.field = field.values
        self.labels = labels

        self._plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 2.0))
        self.layout = QGridLayout()
        self.layout.addWidget(self._plot)
        self.setLayout(self.layout)

        fig = self._plot.getFigure()

        ax = fig.add_subplot(111)
        ax.plot(self.time, self.field)

        ax.set_xlabel(labels['xlabel'])
        ax.set_xlim(self.time[0], self.time[-1])
        ax.xaxis.set_major_locator(dates.SecondLocator(interval=960))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

        ax.set_ylabel(labels['ylabel'])
        ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

        ax.grid(True)

        fig.set_tight_layout(True)


class WaveletPlotWindow(QDialog):
    def __init__(self, time, field, date_format=None, elements=None, labels=None, parent=None):
        super(WaveletPlotWindow, self).__init__(parent)

        self.resize(900, 900)

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName('verticalLayout')

        self.save_cyclotron_button = QPushButton(self)
        self.save_cyclotron_button.setText(_('Save Cyclotron'))
        self.verticalLayout.addWidget(self.save_cyclotron_button)
        self.save_cyclotron_button.clicked.connect(self.save_cyclotron)

        self.integral_value = QLineEdit(self)
        self.verticalLayout.addWidget(self.integral_value)

        self._plot = pg.widgets.MatplotlibWidget.MatplotlibWidget(size=(7.0, 9.0))
        self.verticalLayout.addWidget(self._plot)

        self.time = helpers.get_formatted_time(time, date_format=date_format)
        self.field = field.values
        self.cyclotron_power = {}
        self.elements = elements
        self.labels = labels

        self.open_folder_path = os.path.expanduser(r'~\Documents\me\science\knit')

        fig = self._plot.getFigure()

        self.cyclotron_power, self.integral = wavelet(self.time, self.field, elements=self.elements, labels=self.labels,
                                                      fig=fig)

        self.integral_value.setText(str(self.integral))

    def save_cyclotron(self):
        save_file_filter = 'Text files (*.txt *.asc *.dat *.csv)'
        filename, ok = QFileDialog.getSaveFileName(self, _('Save file'),
                                                   self.open_folder_path,
                                                   filter=save_file_filter)

        self.open_folder_path = os.path.dirname(filename)

        if ok:
            res = {'time': self.time}

            if self.elements:
                for element in self.elements:
                    res[element['name']] = self.cyclotron_power[element['name']]

            result = pd.DataFrame(res)
            result.to_csv(filename, index=False, na_rep=0)


class WaveletAnalysisApp(QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self):
        super(WaveletAnalysisApp, self).__init__()
        self.setupUi(self)

        self.buttonOpenDataFile.clicked.connect(self.open_data)
        self.buttonPlotData.clicked.connect(self.plot_data)
        self.buttonPlotWavelet.clicked.connect(self.plot_wavelet)

        self.data = None
        self.disabled = True

        self.dataModel = PandasModel()
        self.dataView.setModel(self.dataModel)

        self.dataView.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.dataView.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.open_folder_path = os.path.expanduser(r'~\Documents\me\science\knit\data')
        self.open_file_name = None

    def open_data(self):
        open_file_filter = 'Text files (*.txt *.asc *.dat *.csv *.cef);;All files (*.*)'

        filename, ok = QFileDialog.getOpenFileName(self, _('Open file'),
                                                   self.open_folder_path,
                                                   filter=open_file_filter)

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

    def plot_data(self):
        data_range = self.range.text()
        if data_range:
            begin = data_range.split(':')[0]
            end = data_range.split(':')[1]
        else:
            begin = 0
            end = self.data.shape[0]

        time_axis = self.listTime.currentText()
        field_axis = self.listData.currentText()

        time = self.data.loc[begin:end][time_axis]
        field = self.data.loc[begin:end][field_axis]
        date_format = self.dateFormat.currentText()
        labels = {'xlabel': time_axis, 'ylabel': field_axis}

        plt = FieldPlotWindow(time, field, date_format=date_format, labels=labels, parent=self)
        plt.show()

    def plot_wavelet(self):
        data_range = self.range.text()
        if data_range:
            begin = data_range.split(':')[0]
            end = data_range.split(':')[1]
        else:
            begin = 0
            end = self.data.shape[0]

        elements = self.chemical.text()
        elements = elements.split(',')

        elements_mass = {'H': 1, 'He': 4, 'Li': 7, 'Be': 9, 'B': 11, 'C': 12, 'N': 14, 'O': 16, 'F': 19, 'Ne': 20,
                         'Na': 23, 'Mg': 24, 'Al': 27, 'Si': 28, 'P': 30, 'S': 32, 'Cl': 35.5, 'Ar': 40, 'K': 40,
                         'Ca': 40}

        elements_dict = []

        for element in elements:
            if not element:
                continue
            charge = element.count('+')

            element_sym = re.findall('[A-Z][a-z]?', element)[0]

            mass = elements_mass[element_sym]

            elements_dict.append({'name': element, 'mass': mass, 'charge': charge})

        time_axis = self.listTime.currentText()
        field_axis = self.listData.currentText()

        time = self.data.loc[begin:end][time_axis]
        field = self.data.loc[begin:end][field_axis]
        date_format = self.dateFormat.currentText()
        labels = {'title': _('Wavelet Analysis'),
                  'xlabel': _('time, H:M:S'),
                  'ylabel': _('|B|, nT')}

        plt = WaveletPlotWindow(time, field, date_format=date_format, elements=elements_dict, labels=labels,
                                parent=self)
        plt.show()
