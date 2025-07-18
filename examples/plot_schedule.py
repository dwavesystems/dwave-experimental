from pprint import pprint
import numpy
from dwave.experimental import fast_reverse_anneal as fra

solver_name = fra.get_solver_name()
params = fra.get_parameters(solver_name)
schedules = fra.load_schedules(solver_name)

print("Solver:", solver_name)
print("Parameters:")
pprint(params)
print("Schedules:")
pprint(schedules)

t = numpy.arange(0.0, 0.06, 1e-4)
nominal_pause_times = params['x_nominal_pause_time']['limits']['set']

target_c = 0.5
fig = None
for nominal_pause_time in nominal_pause_times:
    fig = fra.plot_schedule(t, target_c, nominal_pause_time,
                            schedules=schedules, figure=fig)

filename = f'schedules for {target_c=}.png'
fig.savefig(filename)
print(f'Figure saved to: {filename}')
