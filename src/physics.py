import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import scipy.interpolate
import scipy.ndimage
import wavelets

from src.helpers import date_formatter, get_from_dict


def cyclotron_frequency(magnetic_field, charge=1, element=1):
    proton_charge = -1.6 * 10 ** -19
    proton_mass = 1.67 * 10 ** -27
    coef = (charge * proton_charge) / (2 * np.pi * element * proton_mass)
    cyclotron_data = np.abs(coef * (magnetic_field * 10 ** -9))
    return cyclotron_data


class Wavelet:
    def __init__(self, t, y):
        self.t = t
        self.y = y

        self.wa = None

    def perform_wavelet_analysis(self):
        same_dx = np.all(np.diff(self.t))

        if not same_dx:
            return

        dt = np.mean(np.diff(self.t))
        self.wa = wavelets.WaveletAnalysis(time=self.t, data=self.y, dt=dt, wavelet=wavelets.Morlet(), unbias=True)

        return self.wa

    def get_cyclotron_on(self, element):
        C, S = self.wa.coi
        t = self.wa.time
        power = self.wa.wavelet_power
        scales = self.wa.scales

        C = np.insert(C, [0, C.size], [t.min(), t.max()])
        S = np.insert(S, [0, S.size], [0, 0])

        scales = np.array([s for s in scales if s < S.max()])
        power = power[0:len(scales)]

        interpolated_coi = scipy.interpolate.interp1d(C, S, bounds_error=False)

        def find_nearest_idx(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx

        charge = element['charge']
        mass = element['mass']

        cyclotron_period = 1 / cyclotron_frequency(self.y, charge, mass)
        cyclotron_period[cyclotron_period > interpolated_coi(t)] = np.nan

        t_ = np.arange(0.0, len(t))
        s_ = np.array([find_nearest_idx(scales, x) for x in cyclotron_period], dtype=float)

        start_idx = np.array([find_nearest_idx(scales, x) for x in 0.1 * cyclotron_period], dtype=float)
        end_idx = np.array([find_nearest_idx(scales, x) for x in 0.4 * cyclotron_period], dtype=float)

        s_[s_ == 0] = np.nan
        cyclotron_power = scipy.ndimage.map_coordinates(power, np.vstack((s_, t_)), order=0)
        cyclotron_power[cyclotron_power == 0] = np.nan

        integral = 0

        for tau in t_:
            tau = int(tau)
            if np.isnan(cyclotron_period[tau]):
                continue

            ss = [power[s, tau] for s in range(int(start_idx[tau]), int(end_idx[tau]))]

            integral += np.nansum(ss) * np.mean(np.diff(self.t)) * cyclotron_period[tau]

        return cyclotron_period, cyclotron_power, integral


class WaveletPlot:
    def __init__(self, time, field, elements=None):
        self.time = time.apply(lambda x: x.tz_localize('utc').timestamp()).values
        self.field = field
        self.elements = elements

    def plot(self, labels=None, wl_params=None, fig=None):
        wat = Wavelet(self.time, self.field)
        wa = wat.perform_wavelet_analysis()
        t = wa.time
        scales = wa.scales
        power = wa.wavelet_power

        if self.elements:
            heights_ratios = [1, 1, 4]
        else:
            heights_ratios = [1, 5]

        height = len(heights_ratios)

        grid = gridspec.GridSpec(height, 1, height_ratios=heights_ratios)

        if self.elements:
            grid.update(wspace=0.0, hspace=0.0)
        ax_magnetic = fig.add_subplot(grid[0])

        formatter = ticker.FuncFormatter(date_formatter)

        ax_magnetic.plot(t, self.field, color='k')

        # ax_magnetic.set_title(labels['title'])

        ax_magnetic.set_xlim(t[0], t[-1])
        ax_magnetic.xaxis.set_major_formatter(formatter)

        ax_magnetic.xaxis.set_visible(False)

        ax_magnetic.set_ylabel(labels['ylabel'])
        ax_magnetic.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

        ax_magnetic.grid(b=True, which='major', color='k', linestyle='dotted', alpha=.2, zorder=3)

        C, S = wa.coi
        S = np.insert(S, [0, S.size], [0, 0])

        scales = np.array([s for s in scales if s < S.max()])
        power = power[0:len(scales)]

        power_min = get_from_dict(wl_params, 'power_min', power.min())
        power_max = get_from_dict(wl_params, 'power_max', power.max())

        c_period = {}
        integral = 0

        if self.elements:
            ax_cyclotron = fig.add_subplot(grid[1], sharex=ax_magnetic)

            for element in self.elements:
                cyclotron_period, cyclotron_power, integral = wat.get_cyclotron_on(element=element)
                c_period[element['name']] = cyclotron_period
                ax_cyclotron.semilogy(t, cyclotron_power)

            ax_cyclotron.set_title(labels['cyclotron_title'])

            ax_cyclotron.set_xlabel(labels['xlabel'])
            ax_cyclotron.set_xlim(t[0], t[-1])
            ax_cyclotron.xaxis.set_major_locator(ticker.AutoLocator())
            ax_cyclotron.xaxis.set_major_formatter(formatter)

            ax_cyclotron.set_ylabel(labels['power'])
            ax_cyclotron.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

            ax_cyclotron.grid(True)

        T, S = np.meshgrid(t, scales)

        ax_wavelet = fig.add_subplot(grid[height - 1], sharex=ax_magnetic)
        levels = np.linspace(power_min, power_max, 64, endpoint=True)
        norm = colors.Normalize(vmin=power_min, vmax=power_max, clip=True)
        s = ax_wavelet.contourf(T, S, power, levels, norm=norm, cmap='coolwarm', extend='max')

        levels = np.linspace(power_min, power_max, 8, endpoint=True)
        s = ax_wavelet.contour(T, S, power, norm=norm, colors='k', extend='max')
        ax_wavelet.clabel(s, fontsize=4, inline=1)

        cb = fig.colorbar(s, ax=ax_wavelet, orientation='horizontal', pad=0.1)
        cb.set_label(labels['power'])
        cb.mappable.set_clim(vmin=power_min, vmax=power_max)

        ax_wavelet.set_xlabel(labels['xlabel'])
        ax_wavelet.set_xlim(t[0], t[-1])
        ax_wavelet.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax_wavelet.xaxis.set_major_formatter(formatter)

        scale_min = get_from_dict(wl_params, 'scale_min', scales.min())
        scale_max = get_from_dict(wl_params, 'scale_max', scales.max())

        ax_wavelet.set_ylabel(labels['scale'])
        ax_wavelet.set_ylim(scale_max, scale_min)
        ax_wavelet.set_yscale('log', nonposy='clip')
        ax_wavelet.set_yticks([10 ** n for n in range(0, 4)])
        ax_wavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())

        C, S = wa.coi
        C = np.insert(C, [0, C.size], [t.min(), t.max()])
        S = np.insert(S, [0, S.size], [0, 0])

        ax_wavelet.fill_between(x=C, y1=S, y2=scale_max, hatch='X', facecolor='none', edgecolor='k')

        if c_period:
            if len(c_period) == 1:
                (_, value), = c_period.items()
                ax_wavelet.fill_between(t, 0.1 * value, 0.4 * value, color='r', alpha=0.5)
            else:
                for key, value in c_period.items():
                    ax_wavelet.plot(t, value, '-', color='r', linewidth=1)

        ax_wavelet.grid(b=True, which='major', color='k', linestyle='dotted', alpha=.2, zorder=3)

        ax_wavelet_fourier = ax_wavelet.twinx()

        ax_wavelet_fourier.set_ylabel(labels['frequency'])
        fourier_lim = [1 / wa.fourier_period(i) for i in ax_wavelet.get_ylim()]
        ax_wavelet_fourier.set_ylim(fourier_lim)
        ax_wavelet_fourier.set_yscale('log')
        ax_wavelet_fourier.set_yticks([10 ** (-n) for n in range(4)])

        fig.set_tight_layout(True)

        return c_period, integral
