# Copyright (C) 2006, 2007 Luigi Ballabio

# This file is part of ControlRoom.

# ControlRoom is free software: you can redistribute it and/or modify it
# under the terms of its license.
# You should have received a copy of the license along with this program;
# if not, please email lballabio@users.sourceforge.net

# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the license for more details.

import os
import math
import logging
import numpy as np
from ControlRoom import *


_torus = Torus(R=1.88*meters, k=1.8)


def bell(peak, rho, alpha, beta, a):
    return a * peak * (1 - rho ** alpha) ** beta + (1 - a) * peak


def rectangle(peak, rho, rho_max=1):
    if rho <= rho_max:
        return peak
    else:
        return 0 * peak


def _valid(line):
    return line.strip() and not line.startswith(r' //')


def _first_valid(file):
    line = file.readline()
    while not _valid(line):
        line = file.readline()
    return line


def _columns(file, transpose=False):
    data = []
    line = _first_valid(file)
    while _valid(line):
        data.append(tuple([float(x) for x in line.strip().split()]))
        line = file.readline()
    if transpose:
        return zip(*data)
    else:
        return data


def _floats(file):
    for line in file:
        for s in line.split():
            yield float(s)


def _fbm_data(file):
    log = logging.getLogger('transp')
    log.info('Reading FBM data...')
    E, dE = zip(*_columns(file))
    mu, dmu = zip(*_columns(file))
    volume = _columns(file)

    inputs = _floats(file)

    data = {}
    for i in range(len(volume)):
        fs = []
        for y in mu:
            for x in E:
                fs.append(inputs.next())
        data[(volume[i][0],volume[i][1])] = TabulatedDistribution(E, mu, fs, eV)
    log.info('done.')

    return E, mu, volume, data


def _Ti_data(file):
    log = logging.getLogger('transp')
    log.info('Reading ion temperature data...')
    data = _columns(file)
    log.info('done.')
    return data


def _Te_data(file):
    if file:
        log = logging.getLogger('transp')
        log.info('Reading electron temperature data...')
        data = _columns(file)
        log.info('done.')
        return data
    else:
        return None


def _n_data(file):
    log = logging.getLogger('transp')
    log.info('Reading density data...')
    data = _columns(file)
    log.info('done.')
    return data


class _Coordinates:
    def __init__(self, file):
        log = logging.getLogger('transp')
        log.info('Reading NUBEAM coordinates...')
        self.data = _columns(file)
        log.info('processing...')
        for i, (x, theta, R, Z) in enumerate(self.data):
            r, _, phi = _torus.toToroidal(R*meters, 0.0*meters, Z*meters)
            self.data[i] = (x,theta,R,Z,float(r/meters),phi)
        xMax = max([x for x, theta, R, Z, r, phi in self.data])
        outer = [(phi, r) for x, theta, R, Z, r, phi in self.data if x == xMax]
        outer.sort()
        outer = [(-1.01*pi,outer[0][1])] + outer + [(1.01*pi,outer[-1][1])]
        self.boundary = LinearInterpolation(*zip(*outer))
        self.rMax = max([r for phi, r in outer])

        n = 200
        m = 720
        self.dr = 1.1*self.rMax/n
        self.dphi = 2*pi/m
        self.volumeMap = list(range(n+1))
        for i in range(n+1):
            r = i*self.dr
            self.volumeMap[i] = list(range(m+1))
            for j in range(m+1):
                phi = -pi+j*self.dphi
                if r < 1.05*self.boundary(phi):
                    self.volumeMap[i][j] = self._realClosestVolume(r, phi)
                else:
                    self.volumeMap[i][j] = None
        log.info('done.')

    def closestVolume(self, r, phi):
        if r >= 1.05*self.boundary(phi):
            return None
        else:
            i = int(r/self.dr)
            j = int((phi+pi)/self.dphi)
            return self.volumeMap[i][j]

    def _realClosestVolume(self, r, phi):
        minD2 = 1e+10
        minV = None
        for x, theta, R, Z, r2, phi2 in self.data:
            d2 = r*r + r2*r2 - 2*r*r2*math.cos(phi-phi2)
            if d2 < minD2:
                minD2 = d2
                minV = (x, theta)
        return minV


class _FbmProfile:
    torus = _torus

    def __init__(self, fbm_data, coordinates):
        self.E, self.mu, self.volume, self.data = fbm_data
        self.coordinates = coordinates

    # profiles to be used by volumeSpectrum
    def distribution(self, r, theta, phi):
        V = self.coordinates.closestVolume(float(r/meters),phi)
        if V is not None:
            return self.data[V]
        else:
            return None

    def density(self, r, theta, phi):
        V = self.coordinates.closestVolume(float(r/meters),phi)
        if V is not None:
            return self.data[V].integral()/centimeters**3
        else:
            return None

    def Te(self, r, theta, phi):
        return None

    def Ti(self, r, theta, phi):
        return None


class _ThermalProfile:
    torus = _torus

    def __init__(self, Ti_core, Te_core, n_core, coordinates):
        self.Ti_core = Ti_core
        self.Te_core = Te_core
        self.n_core = n_core
        self.coordinates = coordinates

    # profiles to be used by volumeSpectrum
    def distribution(self, r, theta, phi):
        T = self.Ti(r, theta, phi)
        if T is not None:
            return Maxwellian(T)
        else:
            return None

    def density(self, r, theta, phi):
        V = self.coordinates.closestVolume(float(r/meters),phi)
        if V is not None:
            x = V[0]
            return bell(self.n_core, x, alpha=2.0, beta=0.5, a=0.9)
        else:
            return None

    def Te(self, r, theta, phi):
        if self.Te_core:
            V = self.coordinates.closestVolume(float(r/meters),phi)
            if V is not None:
                x = V[0]
                return bell(self.Te_core, x, alpha=2.0, beta=1.0, a=0.9)
            else:
                return None
        else:
            return None

    def Ti(self, r, theta, phi):
        V = self.coordinates.closestVolume(float(r/meters), phi)
        if V is not None:
            x = V[0]
            return bell(self.Ti_core, x, alpha=2.0, beta=1.0, a=0.9)
        else:
            return None


class _MonoBeamProfile:
    torus = _torus

    def __init__(self, Eb, mu, n_core, coordinates):
        self.Eb = Eb
        self.mu = mu
        self.n_core = n_core
        self.coordinates = coordinates
        self.pitch_angle = (math.acos(mu) * 180 / math.pi) * degrees
    # profiles to be used by volumeSpectrum

    def distribution(self, r, theta, phi):
        if self.Eb is not None:
            return BeamDistribution(self.Eb, self.pitch_angle)
        else:
            return None

    def density(self, r, theta, phi):
        V = self.coordinates.closestVolume(float(r/meters), phi)
        if V is not None:
            x = V[0]
            return bell(self.n_core, x, alpha=1.7, beta=3.0, a=1.0)
        else:
            return None

    def Te(self, r, theta, phi):
        if self.Eb:
            return self.Eb

    def Ti(self, r, theta, phi):
        if self.Eb:
            return self.Eb


def coordinates(coordinates_file):
    return _Coordinates(open(coordinates_file))


def fbm_profile(fbm_file, coordinates):
    return _FbmProfile(_fbm_data(open(fbm_file)), coordinates)


def thermal_profile(Ti_core, Te_core, n_core, coordinates):
    return _ThermalProfile(Ti_core, Te_core, n_core, coordinates)


def monobeam_profile(Eb, mu, n_core, coordinates):
    return _MonoBeamProfile(Eb, mu, n_core, coordinates)


def gen_dir(time, shot, device):
    dir_shot = os.path.join(time, "%d" % shot)
    dir_sim = os.path.join(dir_shot, "simulation")
    dir_nubeam = os.path.join(dir_sim, "nubeam")
    dir_device = os.path.join(dir_sim, device)
    dir_NES = os.path.join(dir_device, "NES")
    if (not os.path.exists(dir_device)):
        os.mkdir(dir_device)
    if (not os.path.exists(dir_NES)):
        os.mkdir(dir_NES)
    return dir_shot, dir_sim, dir_nubeam, dir_device, dir_NES


reaction = DDN3HeReaction()
whichProduct = 1
Emin, Emax, bins = 500*keV, 3520*keV, 151
Enlist = np.arange(500, 3510, 20)
Randomizer.seed(42)
samples = 100000


def cal_NES(coords_file, fbm_file, los_file, ni, Ti, samples=10**6):
    ni, Ti, Te = ni / meters**3, Ti * keV, Ti * keV
    coords = coordinates(coords_file)
    d_profile = thermal_profile(Ti, Te, ni, coords)
    b_profile = fbm_profile(fbm_file, coords)

    # calculate each component
    pairs = {}
    pairs["thermal"] = (d_profile, d_profile)
    pairs["beam-thermal"] = (d_profile, b_profile)
    pairs["beam-beam"] = (b_profile, b_profile)
    ratio_list = [0.5, 0.5, 0.125]
    direct = np.zeros((len(Enlist), 2))
    direct[:, 0] = Enlist
    spectrums = {}
    for i, (name, pair) in enumerate(pairs.items()):
        log.info('Calculating {} spectrum...'.format(name))
        cells = lineOfSight(losfile, pair[0], pair[1])
        spectrum = CalculateVolumeSpectrum(
            reaction, whichProduct, cells,
            Emin, Emax, bins, samples,
            E1range=(0.0*MeV, 1.0*MeV),
            E2range=(0.0*MeV, 1.2*MeV),
            Bdirection=Clockwise)
        spectrum = spectrum * ratio_list[i]
        data = []
        for En in Enlist:
            data.append([En, spectrum[En * keV].value])
        data = np.array(data)
        spectrums[name] = data
        direct[:, 1] += data[:, 1]
        log.info('done')
    spectrums["direct"] = direct
    return spectrums


def cal_NES_WF(ni, Ti, coords_file, los_file, mulist, Eblist, samples=10**6):

    # set the parameters of thermal plasma and beam ions
    ni, Ti, Te = ni / meters**3, Ti * keV, Ti * keV
    nb = 1.0e+19/meters**3
    coords = coordinates(coords_file)
    d_profile = thermal_profile(Ti, Ti, ni, coords)

    # claculate the neutron energy spectrum for each(Eb, mu) element
    for i, mu in enumerate(mulist):
        for j, Eb in enumerate(Eblist):
            start = time.time()
            log.info("mu: %g, beam energy: %g keV" % (mu, Eb / keV))
            log.info('Calculating beam-thermal spectrum...')
            b_profile = monobeam_profile(Eb*keV, mu, nb, coords)
            cells = lineOfSight(losfile, d_profile, b_profile)
            beamthermal = CalculateVolumeSpectrum(
                reaction, whichProduct, cells,
                Emin, Emax, bins, samples,
                E1range=(0.0*MeV, 1.0*MeV),
                E2range=(0.0*MeV, 1.2*MeV),
                Bdirection=Clockwise)
            log.info('done')
            beamthermal = beamthermal / 2.0
            filename = "mu_%d_Eb_%d" % (i, j)
            data = []
            for En in Enlist:
                data.append([En, beamthermal[En*keV].value])
            data = np.array(data)
            np.savetxt(os.path.join(dir_NES_WF, filename), data, fmt="%g")
            stop = time.time()
            print(stop - start)

    np.savetxt("Enlist", Enlist, fmt="%g")
    np.savetxt("Eblist", Eblist, fmt="%g")
    np.savetxt("mulist", mulist, fmt="%g")
    log.info("number of Eb: %g, number of mu: %g" % (len(Eblist), len(mulist)))


# export spectrums to txt files
for name, spectrum in spectrums.items():
    np.savetxt(os.path.join(dir_NES, name), spectrum, fmt="%g")
    intensity = spectrum[:, 1].sum()
    portion = 100 * intensity / direct[:, 1].sum()
    f_out.write(
        "{}: intensity={} #/s, portion={}\n".format(name, intensity, portion))
f_out.close()



