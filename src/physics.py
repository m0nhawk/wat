import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.ndimage
import wavelets


def cyclotron_frequency(magnetic_field, charge=1, element=1):
    proton_charge = -1.6 * 10 ** -19
    proton_mass = 1.67 * 10 ** -27
    coef = (charge * proton_charge) / (2 * np.pi * element * proton_mass)
    cyclotron_data = np.abs(coef * (magnetic_field * 10 ** -9))
    return cyclotron_data


def wavelet(time, field, elements=None, fig=None, sign=None, wl_params=None):
    dt = time[1] - time[0]
    wa = wavelets.WaveletAnalysis(time=time, data=field, dt=dt, dj=0.125, wavelet=wavelets.Morlet(), unbias=True)
    power = wa.wavelet_power
    scales = wa.scales
    t = wa.time

    if elements:
        heights_ratios = [1, 1, 4]
    else:
        heights_ratios = [1, 5]

    height = len(heights_ratios)

    grid = gridspec.GridSpec(height, 1, height_ratios=heights_ratios)
    grid.update(wspace=0.0, hspace=0.0)
    ax_magnetic = fig.add_subplot(grid[0])

    def date_formatter(x, pos):
        return pd.to_datetime(x, unit='s').strftime('%H:%M:%S')

    formatter = ticker.FuncFormatter(date_formatter)

    ax_magnetic.plot(t, field)

    ax_magnetic.set_title(sign['title'])

    ax_magnetic.set_xlabel('time, UT')
    ax_magnetic.set_xlim(t[0], t[-1])
    ax_magnetic.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_magnetic.xaxis.set_major_formatter(formatter)

    ax_magnetic.xaxis.label.set_visible(False)

    ax_magnetic.set_ylabel(sign['ylabel'])
    ax_magnetic.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

    ax_magnetic.grid(True)

    C, S = wa.coi
    C = np.insert(C, [0, C.size], [t.min(), t.max()])
    S = np.insert(S, [0, S.size], [0, 0])

    scales = np.array([s for s in scales if s < S.max()])
    power = power[0:len(scales)]

    vmin = power.min() if wl_params is None or 'vmin' not in wl_params else wl_params['vmin']
    vmax = power.max() if wl_params is None or 'vmax' not in wl_params else wl_params['vmax']

    interpolated_coi = scipy.interpolate.interp1d(C, S, bounds_error=False)

    def find_nearest_idx(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    c_period = None
    if elements:
        c_period = {}
        c_power = {}

        ax_cyclotron = fig.add_subplot(grid[1])

        for element in elements:
            charge = element['charge']
            mass = element['mass']
            name = element['name']

            cyclotron_period = 1 / cyclotron_frequency(field, charge, mass)
            cyclotron_period[cyclotron_period > interpolated_coi(t)] = np.nan

            c_period[name] = cyclotron_period

            t_ = np.arange(0.0, len(t))
            s_ = np.array([find_nearest_idx(scales, x) for x in cyclotron_period], dtype=float)

            s_[s_ == 0] = np.nan
            cyclotron_power = scipy.ndimage.map_coordinates(power, np.vstack((s_, t_)), order=0)
            cyclotron_power[cyclotron_power == 0] = np.nan

            c_power[name] = cyclotron_power

            ax_cyclotron.semilogy(t, cyclotron_power)

        ax_cyclotron.set_title('Power on cyclotron frequency')

        ax_cyclotron.set_xlabel(sign['xlabel'])
        ax_cyclotron.set_xlim(t[0], t[-1])
        ax_cyclotron.xaxis.set_major_locator(ticker.AutoLocator())
        ax_cyclotron.xaxis.set_major_formatter(formatter)

        ax_cyclotron.set_ylabel('Power, (nT)^2')
        ax_cyclotron.yaxis.set_major_locator(ticker.LinearLocator(numticks=3))

        ax_cyclotron.grid(True)

    T, S = np.meshgrid(t, scales)

    ax_wavelet = fig.add_subplot(grid[height - 1])
    levels = np.linspace(vmin, vmax, 64, endpoint=True)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    s = ax_wavelet.contourf(T, S, power, levels, norm=norm, cmap='coolwarm', extend='max')

    cb = fig.colorbar(s, ax=ax_wavelet, orientation='horizontal', pad=0.1)
    cb.set_label('Wavelet power spectrum (nT)^2')
    cb.set_clim(vmin=vmin, vmax=vmax)

    ax_wavelet.set_xlabel(sign['xlabel'])
    ax_wavelet.set_xlim(t[0], t[-1])
    ax_wavelet.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_wavelet.xaxis.set_major_formatter(formatter)

    scale_min = scales.min() if wl_params is None or 'smin' not in wl_params else wl_params['smin']
    scale_max = scales.max() if wl_params is None or 'smax' not in wl_params else wl_params['smax']

    ax_wavelet.set_ylabel('Scale (s)')
    ax_wavelet.set_ylim(scale_max, scale_min)
    ax_wavelet.set_yscale('log', nonposy='clip')
    ax_wavelet.set_yticks([10 ** n for n in range(2, 3)])
    ax_wavelet.yaxis.set_major_formatter(ticker.ScalarFormatter())

    C, S = wa.coi
    C = np.insert(C, [0, C.size], [t.min(), t.max()])
    S = np.insert(S, [0, S.size], [0, 0])

    ax_wavelet.fill_between(x=C, y1=S, y2=scale_max, color='gray', alpha=0.3)

    if c_period:
        for key, value in c_period.items():
            ax_wavelet.plot(t, value, '-', linewidth=1)

    ax_wavelet.grid(b=True, which='major', color='k', linestyle='-', alpha=.2, zorder=3)

    ax_wavelet_fourier = ax_wavelet.twinx()

    ax_wavelet_fourier.set_ylabel('Frequency (Hz)')
    fourier_lim = [1 / wa.fourier_period(i) for i in ax_wavelet.get_ylim()]
    ax_wavelet_fourier.set_ylim(fourier_lim)
    ax_wavelet_fourier.set_yscale('log')
    ax_wavelet_fourier.set_yticks([10 ** (-n) for n in range(3)])

    fig.set_tight_layout(True)
