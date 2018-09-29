#!/usr/bin/env python2.7
import os
import numpy as np


def inrange(x, range_x):
    """ Function doc """
    x_min = range_x[0]
    x_max = range_x[1]
    idx1 = x > x_min
    idx2 = x < x_max
    return idx1 & idx2


def cash(x_exp, x_sim):
    """ Function doc """
    t1 = x_sim - x_exp
    t2 = x_exp * np.log(x_exp / x_sim)
    return 2 * (t1 + t2).sum()


class PHdata(object):
    """ Class doc """
    @classmethod
    def set_dir_RF(cls, dir_RF):
        """ Function doc """
        cls.dir_RF = dir_RF
        cls.dir_RF_csvdata = os.path.join(dir_RF, "G4csvdata")
        cls.dir_RF_elements = os.path.join(dir_RF, "elements")

    @classmethod
    def FWHM(cls, x, p):
        """
        x: lightout in unit keVee
        a, b, c = p: energy resolution parameters
        a = 4.4, b = 9, c = 0.5 for EJ301
        """
        a, b, c = p
        y = x / 1000.0
        return np.sqrt(a*a + b*b / y + c*c / np.power(y, 2)) / 100.0

    @classmethod
    def cal_RF(cls, binedge, p):
        """
        binedge: the histogram bins
        a, b, c: energy resolution parameters
        """
        Enlist = np.loadtxt(os.path.join(cls.dir_RF, "Enlist"))
        binmiddle = 0.5 * (binedge[:-1] + binedge[1:])
        np.savetxt(os.path.join(cls.dir_RF, "binedge"), binedge, fmt="%g")
        np.savetxt(os.path.join(cls.dir_RF, "binmiddle"), binmiddle, fmt="%g")
        if (not os.path.exists(cls.dir_RF_elements)):
            os.mkdir(cls.dir_RF_elements)
        threshold = 50
        for En in Enlist:
            print("RF at En = %g keV will be calculated" % En)
            file_csv = os.path.join(cls.dir_RF_csvdata, "En_%dkeV" % En)
            ph = np.loadtxt(file_csv)[:, 0]
            ph = ph[ph > threshold]
            fwhm = ph * cls.FWHM(ph, p)
            ph = np.random.normal(ph, fwhm / 2.355)
            histph, bins = np.histogram(ph, bins=binedge)
            file_element = os.path.join(cls.dir_RF_elements, "En_%dkeV" % (En))
            np.savetxt(file_element, histph, fmt="%g")

    @classmethod
    def load_RF(cls, Enlist):
        """ Function doc """
        binedge = np.loadtxt(os.path.join(cls.dir_RF, "binedge"))
        binmiddle = np.loadtxt(os.path.join(cls.dir_RF, "binmiddle"))
        matrix_RF = np.zeros((len(binmiddle), len(Enlist)))
        for i in range(len(Enlist)):
            file_element = os.path.join(
                cls.dir_RF_elements, "En_%dkeV" % Enlist[i])
            matrix_RF[:, i] = np.loadtxt(file_element).reshape(-1, 1)
        return binedge, binmiddle, Enlist, matrix_RF

    # shot data analysis ####
    def __init__(self):
        pass

    def loaddata(self, filename, k=1, b=0):
        """
        filename: the PH data file
        k, b: qlong = k * ph + b
        """
        if not os.path.exists(filename):
            print("%s does not exists" % filename)
            return None
        data = np.loadtxt(filename)
        self.qlong = data[:, 1]
        self.qshort = data[:, 2]
        self.psd = 1 - self.qshort * 1.0 / self.qlong
        self.ph = (self.qlong - b) / k
        # preprocess for timestamp
        T_period = pow(2, 32) - 1
        timestamp = np.zeros(self.qlong.shape)
        T_count = 0
        for i in range(data.shape[0] - 1):
            temp = data[i, 0] + T_count * T_period
            if data[i, 0] > data[i + 1, 0]:
                T_count = T_count + 1
            timestamp[i] = temp
        self.timestamp = timestamp

    def PSD(self, range_ph, range_psd):
        """ Function doc """
        idx_ph = inrange(self.ph, range_ph)
        idx_psd = inrange(self.psd, range_psd)
        return idx_ph & idx_psd


class TOFEDdata(object):
    """ Class doc """
    @classmethod
    def set_dir_RF(cls, dir_RF):
        """ Function doc """
        cls.dir_RF = dir_RF
        cls.dir_RF_csvdata = os.path.join(dir_RF, "G4csvdata")
        cls.dir_RF_elements = os.path.join(dir_RF, "elements")

    @classmethod
    def cal_RF(cls, binedge, fwhm_DAQ=0):
        # daq resolution for s1: 0.5 ns, s2: 0.5 ns, light propagation:
        # 0.6 ns.
        Enlist = np.loadtxt(os.path.join(cls.dir_RF, "Enlist"))
        binmiddle = 0.5 * (binedge[:-1] + binedge[1:])
        np.savetxt(os.path.join(cls.dir_RF, "binedge"), binedge, fmt="%g")
        np.savetxt(os.path.join(cls.dir_RF, "binmiddle"), binmiddle, fmt="%g")
        ph_low = 15
        for En in Enlist:
            print("TOFED RF for En = %g keV will be calculated" % (En))
            file_csv = os.path.join(cls.dir_RF_csvdata, "En_%dkeV" % (En))
            data = np.loadtxt(file_csv)
            tof = data[:, 0]
            ph_s1 = data[:, 1]
            ph_s2 = data[:, 2]
            idx = (ph_s1 > ph_low) & (ph_s2 > ph_low)
            tof = np.random.normal(tof[idx], fwhm_DAQ / 2.355)
            histtof, bins = np.histogram(tof, bins=binedge)
            file_element = os.path.join(cls.dir_RF_elements, "En_%dkeV" % (En))
            np.savetxt(file_element, histtof, fmt="%g")

    @classmethod
    def load_RF(cls, Enlist):
        """ Function doc """
        binedge = np.loadtxt(os.path.join(cls.dir_RF, "binedge"))
        binmiddle = np.loadtxt(os.path.join(cls.dir_RF, "binmiddle"))
        matrix_RF = np.zeros((len(binmiddle), len(Enlist)))
        for i in range(len(Enlist)):
            file_element = os.path.join(
                cls.dir_RF_elements, "En_%dkeV" % Enlist[i])
            matrix_RF[:, i] = np.loadtxt(file_element)
        return binedge, binmiddle, Enlist, matrix_RF

    def __init__(self):
        """ Class initialiser """
        pass

    def set_dir_shot(self, dir_shot):
        """ Function doc """
        self.dir_shot = dir_shot

    def set_DGZ_type(self, DGZ_type):
        """ Function doc """
        self.DGZ_type = DGZ_type
        self.dir_tof = os.path.join(self.dir_shot, self.DGZ_type + "_toflist")

    def load_divergence(self, filename):
        """ Function doc """
        self.divergence = {}
        data = np.loadtxt(filename)
        numpair = data.shape[0]
        for i in range(numpair):
            pair = (data[i, 0], data[i, 1])
            value = data[i, 2]
            self.divergence[pair] = value
        return self.divergence

    def cal_board_offset(self, reflist):
        """ Function doc """
        self.board_offset = [0]
        binedge = np.arange(-30, 30, 0.1)
        binmiddle = 0.5 * (binedge[1:] + binedge[:-1])
        for i in range(1, len(reflist)):
            ch1 = reflist[0]
            ch2 = reflist[i]
            filename = os.path.join(self.dir_tof, "%d_%d" % (ch1, ch2))
            filesize = os.path.getsize(filename)
            if filesize > 0:
                data = np.loadtxt(filename)
                (ntof, bins) = np.histogram(data[:, 0], bins=binedge)
                idx_ymax, =  np.where(ntof == ntof.max())
                self.board_offset.append(binmiddle[idx_ymax].mean())
        filename = os.path.join(self.dir_tof, "board_offset")
        np.savetxt(filename, self.board_offset, fmt="%g")

    def load_board_offset(self):
        """ Function doc """
        filename = os.path.join(self.dir_tof, "board_offset")
        self.board_offset = np.loadtxt(filename)
        return self.board_offset

    def load_pair(self, ch1, ch2, include_divergence=False):
        """ Function doc """
        filename = os.path.join(self.dir_tof, "%d_%d" % (ch1, ch2))
        if not os.path.exists(filename):
            return None
        if os.path.getsize(filename) == 0:
            return None
        data = np.loadtxt(filename).reshape(-1, 4)
        offset = self.board_offset[int(ch1/8)] - self.board_offset[int(ch2/8)]
        if include_divergence:
            offset = offset + self.divergence[(ch1, ch2)]
        data[:, 0] = data[:, 0] + offset
        return data

    def load_pairs(self, ch1_list, ch2_list, include_divergence=False):
        """ Function doc """
        data_pairs = np.zeros((1, 4))
        for ch1 in ch1_list:
            for ch2 in ch2_list:
                data = self.load_pair(ch1, ch2, include_divergence)
                if data is not None:
                    data_pairs = np.vstack((data_pairs, data))
        return data_pairs
