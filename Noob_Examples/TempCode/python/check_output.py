import numpy as np
import matplotlib.pyplot as plt

path = "."
fspc = open(path + "/spc.txt", 'r')
fspcFFT = open(path + "/spcFFT.txt", 'r')

nan = np.nan


def FFT_of_time_to_puls_domain(Z, dt=1 / 50., opt=0):
    N = Z.shape[0]

    Om = 2 * np.pi * np.fft.fftfreq(N, dt)

    idPos = Om >= 0

    domFFT = np.mean(np.diff(Om[:N // 2]))

    if opt == 0:
        A = Z / (2 * domFFT)  #
    else:
        Z = np.fft.fft(Z, N)
        A = (2 * np.abs(Z) / N) ** 2 / (2 * domFFT)

    # A = Z# / (2*domFFT)

    plt.plot(Om, A)


# plt.show()


try:
    data1 = f.read().split(",\n")
    data2 = [eval(d) for d in data1 if d]
    data3 = [np.complex(d[0], d[1]) for d in data2]
    data4 = np.abs(np.r_[data3])
    opt = 1


except:
    z = np.fromfile(path + "/z.txt", float, -1, ",\n")
    z1 = np.fromfile(path + "/z1.txt", float, -1, ",\n")
    om = np.fromfile(path + "/om.txt", float, -1, ",\n")
    spc = np.fromfile(path + "/spc.txt", float, -1, ",\n")
    omfft = np.fromfile(path + "/omFFT.txt", float, -1, ",\n") + 1
    spcfft = np.fromfile(path + "/spcFFT.txt", float, -1, ",\n")

# N = spcfft.shape[0]*2#z.shape[0]*2
# dt = 1/50.
# Om = 2 * np.pi * np.fft.fftfreq( N, dt )

# idPos = Om >= 0

# domFFT = np.mean( np.diff( Om[:N // 2 ] ) )

# spcfft /= domFFT

# FFT_of_time_to_puls_domain(z,opt=1);

plt.figure(1)
plt.plot(z1)
plt.plot(z)
plt.show(block=False)

plt.figure(2)
plt.plot(om, spc)
plt.plot(omfft, spcfft)
plt.show(block=True)  # plt.plot(data4)
# plt.show()
