# Verifies that the sum of the power
# in the reflected and transmitted
# orders of a 1D binary grating
# is equivalent to the power
# of the input planewave source.


import unittest
import parameterized
import meep as mp
import math
import cmath
import numpy as np


class TestEigCoeffs(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.resolution = 100   # pixels/μm

    cls.dpml = 2.0         # PML thickness
    cls.dsub = 3.0         # substrate thickness
    cls.dpad = 3.0         # padding thickness between grating and PML
    cls.gp = 2.3           # grating period
    cls.gh = 0.5           # grating height
    cls.gdc = 0.5          # grating duty cycle

    cls.sx = cls.dpml+cls.dsub+cls.gh+cls.dpad+cls.dpml
    cls.sy = cls.gp

    cls.cell_size = mp.Vector3(cls.sx,cls.sy,0)

    cls.pml_layers = [mp.PML(thickness=cls.dpml,direction=mp.X)]

    wvl = 0.5              # center wavelength
    cls.fcen = 1/wvl       # center frequency
    cls.df = 0.05*cls.fcen # frequency width

    cls.ng = 1.5
    cls.glass = mp.Medium(index=cls.ng)

    cls.geometry = [mp.Block(material=cls.glass,
                             size=mp.Vector3(cls.dpml+cls.dsub,mp.inf,mp.inf),
                             center=mp.Vector3(-0.5*cls.sx+0.5*(cls.dpml+cls.dsub),0,0)),
                    mp.Block(material=cls.glass,
                             size=mp.Vector3(cls.gh,cls.gdc*cls.gp,mp.inf),
                             center=mp.Vector3(-0.5*cls.sx+cls.dpml+cls.dsub+0.5*cls.gh,0,0))]


  @parameterized.parameterized.expand([(0.,), (10.7,)])
  def test_binary_grating_oblique(self, theta):
    # rotation angle of incident planewave
    # counterclockwise (CCW) about Z axis, 0 degrees along +X axis
    theta_in = math.radians(theta)

    src_cmpt = mp.Hz # mp.Ez (S pol.) or mp.Hz (P pol.)

    if theta_in == 0:
      symmetries = [mp.Mirror(mp.Y,phase=+1 if src_cmpt == mp.Ez else -1)]
      k = mp.Vector3()
    else:
      symmetries = []
      # k (in source medium) with correct length (plane of incidence: XY)
      k = mp.Vector3(self.fcen*self.ng).rotate(mp.Vector3(0,0,1), theta_in)

    def pw_amp(k,x0):
      def _pw_amp(x):
        return cmath.exp(1j*2*math.pi*k.dot(x+x0))
      return _pw_amp

    src_pt = mp.Vector3(-0.5*self.sx+self.dpml,0,0)
    sources = [mp.Source(mp.GaussianSource(self.fcen,fwidth=self.df),
                         component=src_cmpt,
                         center=src_pt,
                         size=mp.Vector3(0,self.sy,0),
                         amp_func=pw_amp(k,src_pt))]

    sim = mp.Simulation(resolution=self.resolution,
                        cell_size=self.cell_size,
                        boundary_layers=self.pml_layers,
                        k_point=k,
                        default_material=self.glass,
                        sources=sources,
                        symmetries=symmetries)

    refl_pt = mp.Vector3(-0.5*self.sx+self.dpml+0.5*self.dsub,0,0)
    refl_flux = sim.add_mode_monitor(self.fcen,
                                     0,
                                     1,
                                     mp.FluxRegion(center=refl_pt,
                                                   size=mp.Vector3(0,self.sy,0)))

    stop_cond = mp.stop_when_fields_decayed(50, src_cmpt, refl_pt, 1e-9)
    sim.run(until_after_sources=stop_cond)

    input_flux = mp.get_fluxes(refl_flux)
    input_flux_data = sim.get_flux_data(refl_flux)

    sim.reset_meep()

    sim = mp.Simulation(resolution=self.resolution,
                        cell_size=self.cell_size,
                        boundary_layers=self.pml_layers,
                        geometry=self.geometry,
                        k_point=k,
                        sources=sources,
                        symmetries=symmetries)

    refl_flux = sim.add_mode_monitor(self.fcen,
                                     0,
                                     1,
                                     mp.FluxRegion(center=refl_pt,
                                                   size=mp.Vector3(0,self.sy,0)))

    sim.load_minus_flux_data(refl_flux,input_flux_data)

    tran_pt = mp.Vector3(0.5*self.sx-self.dpml,0,0)
    tran_flux = sim.add_mode_monitor(self.fcen,
                                     0,
                                     1,
                                     mp.FluxRegion(center=tran_pt,
                                                   size=mp.Vector3(0,self.sy,0)))

    sim.run(until_after_sources=stop_cond)

    # number of reflected orders in substrate
    m_plus = int(np.floor((self.fcen*self.ng-k.y)*self.gp))
    m_minus = int(np.ceil((-self.fcen*self.ng-k.y)*self.gp))

    if theta == 0:
      orders = range(m_plus+1)
    else:
      orders = range(m_minus,m_plus+1)

    Rsum = 0
    for nm in orders:
      res = sim.get_eigenmode_coefficients(refl_flux,
                                           mp.DiffractedPlanewave((0,nm,0),
                                                                  mp.Vector3(0,1,0),
                                                                  1 if src_cmpt == mp.Ez else 0,
                                                                  0 if src_cmpt == mp.Ez else 1))
      r_coeffs = res.alpha
      R = abs(r_coeffs[0,0,1])**2/input_flux[0]
      print(f"refl-order:, {nm:+d}, {R:.6f}")
      Rsum += 2*R if (theta == 0 and nm != 0) else R

    # number of transmitted orders in air
    m_plus = int(np.floor((self.fcen-k.y)*self.gp))
    m_minus = int(np.ceil((-self.fcen-k.y)*self.gp))

    if theta == 0:
      orders = range(m_plus+1)
    else:
      orders = range(m_minus,m_plus+1)

    Tsum = 0
    for nm in orders:
      res = sim.get_eigenmode_coefficients(tran_flux,
                                           mp.DiffractedPlanewave((0,nm,0),
                                                                  mp.Vector3(0,1,0),
                                                                  1 if src_cmpt == mp.Ez else 0,
                                                                  0 if src_cmpt == mp.Ez else 1))
      t_coeffs = res.alpha
      T = abs(t_coeffs[0,0,0])**2/input_flux[0]
      print(f"tran-order:, {nm:+d}, {T:.6f}")
      Tsum += 2*T if (theta == 0 and nm != 0) else T

    r_flux = mp.get_fluxes(refl_flux)
    t_flux = mp.get_fluxes(tran_flux)
    Rflux = -r_flux[0]/input_flux[0]
    Tflux =  t_flux[0]/input_flux[0]

    print(f"refl:, {Rsum:.6f}, {Rflux:.6f}")
    print(f"tran:, {Tsum:.6f}, {Tflux:.6f}")
    print(f"sum:,  {Rsum+Tsum:.6f}, {Rflux+Tflux:.6f}")

    self.assertAlmostEqual(Rsum,Rflux,places=2)
    self.assertAlmostEqual(Tsum,Tflux,places=2)
    self.assertAlmostEqual(Rsum+Tsum,1.00,places=2)


if __name__ == '__main__':
  unittest.main()
