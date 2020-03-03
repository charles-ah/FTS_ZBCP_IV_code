## INITIALIZATION STUFF

import labrad
import sweeper
import time
cxn = labrad.connect()

lck = cxn.sr830()
lck.select_device()

temp = cxn.lakeshore_372
temp.select_device()

mag = cxn.ami_430
mag.select_device()

dc = cxn.dcbox_quad_ad5780
dc.select_device()

ls = cxn.lakeshore_372
ls.select_device()


# record lockin and temperature at a fixed interval

reload(sweeper)
swp = sweeper.Sweeper()

# swp.add_axis(-0.55,0.25,500)
swp.add_axis(-1,1,1000)
swp.add_swept_setting('dev', label="V_BG [V]", setting=['dcbox_quad_ad5780',None,'set_voltage'], inputs=[1], var_slot=1, max_ramp_speed=0.5)
swp.add_recorded_setting('dev', label="Lockin X [A]", setting=['sr830',None,'x'],inputs=[])
swp.add_recorded_setting('dev', label="Lockin Y [A]", setting=['sr830',None,'y'],inputs=[])
#swp.add_recorded_setting('dev', label="T_MC [K]", setting=['lakeshore_372',None,'mc'],inputs=[])
#swp.add_recorded_setting('dev', label="T_PROBE [K]", setting=['lakeshore_372',None,'probe'],inputs=[])

#NAME,UNITS,VALUE


swp.generate_mesh([[0, 1]])
swp.initalize_dataset('TEST', '\\TEST\\')
swp.add_parameters([['Lockin Tau','s',lck.time_constant()],['Lockin Sine Out Amplitude','V_rms',lck.sine_out_amplitude()],
				   ['T_MC','K',float(ls.mc())],['T_PROBE','K',float(ls.probe())],['B','T',float(mag.get_field_mag())],
					['R_BIAS','Ohms',100e6]])



swp.autosweep(stepsize=0.5)


swp = sweeper.Sweeper()

# swp.add_axis(-0.55,0.25,500)
swp.add_axis(-0.2,0.2,5000)
swp.add_swept_setting('dev', label="V_BG [V]", setting=['dcbox_quad_ad5780',None,'set_voltage'], inputs=[1], var_slot=1, max_ramp_speed=0.05)
swp.add_recorded_setting('dev', label="Lockin X [A]", setting=['sr830',None,'x'],inputs=[])
swp.add_recorded_setting('dev', label="Lockin Y [A]", setting=['sr830',None,'y'],inputs=[])
#swp.add_recorded_setting('dev', label="T_MC [K]", setting=['lakeshore_372',None,'mc'],inputs=[])
#swp.add_recorded_setting('dev', label="T_PROBE [K]", setting=['lakeshore_372',None,'probe'],inputs=[])

#NAME,UNITS,VALUE


swp.generate_mesh([[0, 1]])
swp.initalize_dataset('RvG_Vbias_14T_rot0125', '\\EMS006\\')
swp.add_parameters([['Lockin Tau','s',lck.time_constant()],['Lockin Sine Out Amplitude','V_rms',lck.sine_out_amplitude()],
				   ['T_MC','K',float(ls.mc())],['T_PROBE','K',float(ls.probe())],['B','T',float(mag.get_field_mag())],
					['V_DIVIDER','',1000],['I_GAIN','A/V',1e-8]])


dc.set_voltage(1 , -0.2)
time.sleep(5)
swp.autosweep(stepsize=0.5)


##record resistance as we cool

reload(sweeper)
swp = sweeper.Sweeper()

# swp.add_axis(-0.55,0.25,500)
wait_time = 30
total_time = 30*60*60
total_pts = total_time/wait_time


swp.add_axis(0,total_pts,total_pts)
# swp.add_swept_setting('dev', label="step num", setting=['dcbox_quad_ad5780',None,'do_nothing'], inputs=[], var_slot=[0], max_ramp_speed=100)
swp.add_recorded_setting('dev', label="T_PROBE [K]", setting=['lakeshore_372',None,'probe'],inputs=[])
swp.add_recorded_setting('dev', label="T_MC [K]", setting=['lakeshore_372',None,'mc'],inputs=[])
swp.add_recorded_setting('dev', label="Lockin X [A]", setting=['sr830',None,'x'],inputs=[])
swp.add_recorded_setting('dev', label="Lockin Y [A]", setting=['sr830',None,'y'],inputs=[])

swp.generate_mesh([])
swp.initalize_dataset('RvT', '\\Sr3Ir2O7_A\\')
swp.add_parameters([['Lockin Tau','s',lck.time_constant()],['Lockin Sine Out Amplitude','V_rms',lck.sine_out_amplitude()],
					['B','T',float(mag.get_field_mag())],['R_BIAS','Ohms',1e6]])

while ~swp.done():
	swp.step()
	time.sleep(wait_time)



## sweeping IV


reload(sweeper)


for i in range(1,200):
	swp = sweeper.Sweeper()
	swp.add_axis(-5,5,10)

	swp.add_swept_setting('dev', label="Applied voltage", setting=['dcbox_quad_ad5780',None,'set_voltage'], inputs=[1], var_slot=1, max_ramp_speed=50)
	swp.add_recorded_setting('dev', label="I []", setting=['dac_adc',None,'read_voltage'],inputs=[1])
	swp.add_recorded_setting('dev', label="V []", setting=['dac_adc',None,'read_voltage'],inputs=[0])

	swp.generate_mesh([[0, 1]])

	swp.add_parameters([['Probe T','K', float(temp.probe())],['MC T','K',float(temp.mc())],
						['B','T',float(mag.get_field_mag())],['R_BIAS','Ohms',100e3] ,['V_gain','',1e4], ['I_gain','',1e-5]])

	swp.initalize_dataset('IV', '\\Sr3Ir2O7_A\\Tdep_DC')


	dc.set_voltage(1,-5)
	time.sleep(10)

	while not swp.done():
		swp.step()
		time.sleep(3)


## run aa single IV curve


swp = sweeper.Sweeper()
swp.add_axis(-0.5, 0.5, 50)

swp.add_swept_setting('dev', label="Applied voltage", setting=['dcbox_quad_ad5780',None,'set_voltage'], inputs=[1], var_slot=1, max_ramp_speed=50)
swp.add_recorded_setting('dev', label="I []", setting=['dac_adc',None,'read_voltage'],inputs=[1])
swp.add_recorded_setting('dev', label="V []", setting=['dac_adc',None,'read_voltage'],inputs=[0])

swp.generate_mesh([[0, 1]])

swp.add_parameters([['Probe T','K', float(temp.probe())],['MC T','K',float(temp.mc())],
					['B','T',float(mag.get_field_mag())],['R_BIAS','Ohms',100e3] ,['V_gain','',1e4], ['I_gain','',1e-5]])

swp.initalize_dataset('Vbias_highT', '\\EMS006')


dc.set_voltage(1,-0.5)
time.sleep(10)

while not swp.done():
	swp.step()
	time.sleep(3)




## 2D voltage sweep

import labrad
import sweeper
import time
cxn = labrad.connect()

lck = cxn.sr830()
lck.select_device()

temp = cxn.lakeshore_372
temp.select_device()

mag = cxn.ami_430
mag.select_device()

dc = cxn.dcbox_quad_ad5780
dc.select_device()

ls = cxn.lakeshore_372
ls.select_device()


# record lockin and temperature at a fixed interval

reload(sweeper)
swp = sweeper.Sweeper()

# swp.add_axis(-0.55,0.25,500)
swp.add_axis(-0.5,0.5,100)
swp.add_axis(-5,5,20)
swp.add_swept_setting('dev', label="V_BG [V]", setting=['dcbox_quad_ad5780',None,'set_voltage'], inputs=[1], var_slot=1, max_ramp_speed=0.1)
swp.add_swept_setting('dev', label="V_TG [V]", setting=['dcbox_quad_ad5780',None,'set_voltage'], inputs=[2], var_slot=1, max_ramp_speed=0.1)
swp.add_recorded_setting('dev', label="Lockin X [A]", setting=['sr830',None,'x'],inputs=[])
swp.add_recorded_setting('dev', label="Lockin Y [A]", setting=['sr830',None,'y'],inputs=[])

swp.generate_mesh([[0, 1,0], [0,0,1]])
swp.initalize_dataset('RvG_67', '\\HZS58\\')
swp.add_parameters([['Lockin Tau','s',lck.time_constant()],['Lockin Sine Out Amplitude','V_rms',lck.sine_out_amplitude()],
				   ['T_MC','K',float(ls.mc())],['T_PROBE','K',float(ls.probe())],['B','T',float(mag.get_field_mag())],
					['R_BIAS','Ohms',100e6]])


swp.autosweep(stepsize=0.1)
