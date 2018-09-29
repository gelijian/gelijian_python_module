import numpy as np
from scipy import interpolate


class manager(object):
    """
    This is a class for anaylis the result of the G4
    model of TOFED.
    """

    def __init__(self, file_lop_s1, file_lop_s2, radius=0.75):

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
        self.radius = 0.75  # meters

        # load light output data
        data_lop_s1 = np.loadtxt(file_lop_s1)
        self.lop_s1 = interpolate.interp1d(
            data_lop_s1[:, 0], data_lop_s1[:, 1])
        data_lop_s2 = np.loadtxt(file_lop_s2)
        self.lop_s2 = interpolate.interp1d(
            data_lop_s2[:, 0], data_lop_s2[:, 1])

    def get_En(self, tof):

        """
        args:
            tof: time of flight of the neutron (unit: ns)
            flight_distance: the flight distance of the neutron(unit: meter)
        return:
            the energy of the neutron(unit keV)
        """
        constant = 5228.157
        En = constant * ((2 * self.radius) ** 2) / tof ** 2
        return En * 1000

    def ph_s1(self, tof, angle):
        """
        args:
            En: neutron energy keV
            angle: scattering angle
        return:
            the pulse height (in unit keVee) of the proton generated in S1.
        """
        Ep = self.get_En(tof) * np.sin(np.radians(angle)) ** 2
        return self.lop_s1(Ep)

    def ph_s2(self, tof, angle):
        """
        args:
            En: neutron energy keV
            angle: scattering angle
        return:
            the maximum pulse height (in unit keVee)
            of the proton generated in S2.
        """
        Ep = self.get_En(tof) * np.cos(np.radians(angle)) ** 2
        return self.lop_s2(Ep)

    def fix_sel(self, ph, range_ph):
        """
        args:
            En: neutron energy
            k: make the threshold wider
        return:
            bool array
            True: if the event meet the condition.
            False: if the event does ont meet the condition.
        """
        # selection on S1, Eees1 should be in the fixed window
        ph_low = range_ph[0]
        ph_high = range_ph[1]
        idx1 = (ph > ph_low)
        idx2 = (ph < ph_high)
        return idx1 & idx2

    def kin_sel_s1(self, ph, tof, range_angle, k=[1.0, 1.0]):
        """
        single kinematics energy selection
        args:
            ph: ph list of s1
            tof: time of flight
            range_angle: flight angle range of scaterring neutrons
            for TOFED it is (18, 42) degrees with no regards of the two rings.
        return:
            bool array
        """
        ph_low = self.ph_s1(tof, range_angle[0]) * k[0]
        ph_high = self.ph_s1(tof, range_angle[1]) * k[1]
        idx1 = ph > ph_low
        idx2 = ph < ph_high
        return idx1 & idx2

    def kin_sel_s2(self, ph, tof, angle_min, k=1.0):
        """
        single kinematics energy selection
        args:
            ph: ph list of s1
            tof: time of flight
            angle_min: the minimum of flight angle
            for TOFED it is (18, 42) degrees with no regards of the two rings.
        return:
            bool array
        """
        ph_high = self.ph_s1(tof, angle_min) * k
        return ph < ph_high
