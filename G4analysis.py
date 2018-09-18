import numpy as np
from scipy import interpolate


class manager(object):
    """
    This is a class for anaylis the result of the G4
    model of TOFED.
    """

    def __init__(
                self, filename, protonrespfile, S1ID_list,
                S2ID_list, flight_angle_range, th_tof=20, th_DAQ=0):

        """
        args:
            filename: the name of the G4 results(csv file).
                the format is like below:
                tof, Eee_s1, Eee_s2, S1_ID, S2_ID, eventID
                S2_up(S2_ID < 40) S2_bottom(S2_ID >= 40)
            protonrespfile: the proton response function
                Ep(energy of the proton(keV))
                Eee(keV)
            S1ID_list: the detector ID of S1 detector
            S2ID_list: the detector ID of S2 detector
        """
        # geometry
        self.TOFradius = 0.75  # meters
        self.flight_angle_range = flight_angle_range  # degrees
        # load data from pronton response file
        ProtonRespdata = np.loadtxt(protonrespfile)
        self.Eplist = ProtonRespdata[:, 0]
        self.Eeelist = ProtonRespdata[:, 1]
        self.protonrespfun = interpolate.interp1d(self.Eplist, self.Eeelist)
        # load data from G4 file
        data = np.loadtxt(filename)
        index_DAQ = (data[:, 1] > th_DAQ) & (data[:, 2] > th_DAQ)
        index_tof = data[:, 0] > th_tof
        index_S1 = np.in1d(data[:, 3], S1ID_list)
        index_S2 = np.in1d(data[:, 4], S2ID_list)
        index = index_DAQ & index_tof & index_S1 & index_S2
        self.filename = filename
        self.data = data[index, :]
        self.tof = self.data[:, 0]
        self.Eees1 = self.data[:, 1]
        self.Eees2 = self.data[:, 2]
        self.S1_ID = self.data[:, 3]
        self.S2_ID = self.data[:, 4]
        self.eventID = self.data[:, 5]
        self.Enlist = self.Get_En(self.tof)
        self.th_DAQ = th_DAQ
        print("%s has loaded successfully!" % (self.filename))

    def Get_En(self, tof):

        """
        args:
            tof: time of flight of the neutron (unit: ns)
            flight_distance: the flight distance of the neutron(unit: meter)
        return:
            the energy of the neutron(unit keV)
        """
        constant = 5228.157
        En = constant * ((2 * self.TOFradius) ** 2) / tof ** 2
        return En * 1000

    def PHofS1(self, En, flight_angle):
        """
        args:
            En: neutron energy keV
            flight_angle: flght angle
        return:
            the pulse height(Eee) of the proton generated in S1 detector.
            (the scatter neutron will arrive the S2 detector)
        """
        Ep = En * (np.sin(flight_angle / 180.0 * np.pi) ** 2)
        return self.protonrespfun(Ep)

    def PHofS2(self, En, flight_angle):
        """
        args:
            En: neutron energy keV
            flight_angle: flght angle
        return:
            the max pulse height(Eee) of the proton generated in S2 detector.
            (the scatter neutron will arrive the S2 detector)
        """
        Ep = En * (np.cos(flight_angle / 180.0 * np.pi) ** 2)
        print(Ep)
        return self.protonrespfun(Ep)

    def fixed_selection_s1(self, En, ratio_wider=0.0):
        """
        args:
            En: neutron energy
            ratio_expand: make the threshold wider
        return:
            bool array
            it is True if the event meet the condition.
            it is False if the event does ont meet the condition.
        """
        # selection on S1, Eees1 should be in the fixed window
        th_min = self.PHofS1(En, self.flight_angle_range[0]) * (1 - ratio_wider)
        th_max = self.PHofS1(En, self.flight_angle_range[1]) * (1 + ratio_wider)
        return (self.Eees1 > th_min) & (self.Eees1 < th_max)

    def kinetic_selection_s1(self, ratio_wider=0.0):
        """
        single kinematics energy selection
        args:
            flight_angle_range: neutron flight angle range of the s2 detector
            for TOFED it is (18, 42) degrees with no regards of the two rings.
        return:
            bool array
        """
        th_min = self.PHofS1(self.Enlist, self.flight_angle_range[0]) * (1 - ratio_wider)
        th_max = self.PHofS1(self.Enlist, self.flight_angle_range[1]) * (1 + ratio_wider)
        return (self.Eees1 > th_min) & (self.Eees1 < th_max)

    def kinetic_selection_s2(self, ratio_wider=0.0):
        th_max = self.PHofS2(self.Enlist, self.flight_angle_range[0]) * (1 + ratio_wider)
        return (self.Eees2 > self.th_DAQ) & (self.Eees2 < th_max)
