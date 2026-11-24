import matplotlib.pyplot as plt
from brian2 import *
prefs.codegen.target = "numpy"


#cnstant of the stdp function.
tau_pre = 20*ms
tau_post = 20*ms

#magnitude/weight of pre-synaptic neuron (this is the size of the spike)
A_pre = 0.01

# magnitude of the spike in post synaptic neuron
A_post = -A_pre*1.05

#the difference between t_pre and t_post (the time each of them is spiking)
# a list of 100 values between -50 and 50 ms
# each value represents a specific time difference between the pre-synaptic
# and post-synaptic neuron firings.
# its t_post - t_pre
delta_t = linspace(-50, 50, 100)*ms


#the stdp function from the NumPy library, used to calculate the weight (W)
# of the synaptic connection between the pre-synaptic and post-synaptic
# neurons based on the time difference (delta_t) and the given conditions.

W = where(delta_t > 0, A_pre*exp(-delta_t/tau_pre), A_post*exp(delta_t/tau_post))

#plotting the weight (W) of the synaptic connection between the pre-synaptic
plot(delta_t/ms, W)
xlabel(r'$\Delta t$ (ms)')
ylabel('W')
axhline(0, ls='-', c='k')
plt.show()


# if the pre_synaptic is spiking before the post_synaptic neuron,
# then the weight is positive, otherwise it is negative.


