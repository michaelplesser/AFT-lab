import numpy as np
import matplotlib.pyplot as plt
import math

def FFT(x, y):
    ''' 
        Perform the Fast Fourier Transform
    ''' 
    ft    = np.fft.fft(y)
    mag   = np.abs(ft)                                   # Perform the FFT
    freqs = np.linspace(0, 1./(x[1]-x[0]), len(x))       # Generate the frequency data (transformed x-values)
    ## This is a rather subtle point related to fourier transforms
    ## Since the input is real-valued, only half of the output is actual info.
    ft    = ft[   :len(freqs)//2]
    mag   = mag[  :len(freqs)//2]
    freqs = freqs[:len(freqs)//2]
    return freqs, ft, mag

#def clean_data(x, y):
#    x_cleaned, y_cleaned = np.array([]), np.array([])
#    for xi, yi in zip(x, y):
#        if not xi in x_cleaned:
#            x_cleaned = np.append(x_cleaned, xi)
#            y_avg = 0
#            idxs = np.where(xi==x_cleaned) 
#            for j in idxs:
#                y_avg += y[j]
#            y_avg /= len(idxs)
#            y_cleaned = np.append(y_cleaned, y_avg)
#    return x_cleaned, y_cleaned

def main():

    def PartA():
        print("#"*40)
        print("Part A:")
        print("#"*10)

        def plot_n_fft(t):
            n = 2048
            x = np.linspace(0, t, num=n)
            y = np.sin(2*np.pi*100*x)
            plt.plot(x, y)
            plt.show()

            window = np.hanning(n)
        
            y = np.multiply(y, window)
            freqs, fft, ft = FFT(x, y)
            plt.plot(freqs, ft)
            plt.xlim([0, 200])
            plt.show()
            imax = np.where(ft == max(ft))[0]
            FWHM = (freqs[imax+1]-freqs[imax-1])/2.
            fullrange = np.where(ft > 0.1)
            freqrange = freqs[fullrange[0][-1]]-freqs[fullrange[0][0]]
            print("FWHM = %s Hz" % FWHM[0])
            print("Full freq. range = %s Hz" % freqrange)
            print("Cycles in range: %s" %(100.*t))
            return x, y, freqs, fft

        xs, ys, fs, ffts = plot_n_fft(0.1)
        xl, yl, fl, fftl = plot_n_fft(1)
        
        plt.plot(fs, np.abs(ffts))
        plt.plot(fl, np.abs(fftl))
        plt.xlim([0, 200])
        plt.show()

    def PartB():
        print("#"*40)
        print("Part B:")
        print("#"*10)
        d_sin    = np.loadtxt('partb_sin.csv'   , delimiter=',', unpack=True)
        d_square = np.loadtxt('partb_square.csv', delimiter=',', unpack=True)
        
        n = 2048
        x_sin, y_sin = d_sin[0][:n], d_sin[1][:n]
        x_square, y_square = d_square[0][:n], d_square[1][:n]
        window = np.hanning(n)

        y_sin = np.multiply(y_sin, window)
        y_square = np.multiply(y_square, window)

        freqs_sin, fft_sin, ft_sin = FFT(x_sin, y_sin)
        freqs_square, fft_square, ft_square = FFT(x_square, y_square)

        plt.plot(freqs_sin, ft_sin)
        plt.plot(freqs_square, ft_square)
        plt.xlim([0, 10000])
        plt.show()

    def PartC():
        print("#"*40)
        print("Part C:")
        print("#"*10)
        d_fork = np.loadtxt('partc_fork.csv', delimiter=',', unpack=True)
        
        n = 2048
        window = np.hanning(n)
        x_fork, y_fork = d_fork[0][:n], d_fork[1][:n]
        y_fork = np.multiply(y_fork, window)
        
        freqs_fork, fft_fork, ft_fork = FFT(x_fork, y_fork)

        plt.plot(freqs_fork, ft_fork)
        plt.show()
        
        d_beat = np.loadtxt('partc_beat.csv', delimiter=',', unpack=True)
        x_beat, y_beat = d_beat[0][:n], d_beat[1][:n]

        plt.plot(x_beat, y_beat)
        plt.show()

    def PartD():
        print("#"*40)
        print("Part D:")
        print("#"*10)
        d_justin  = np.loadtxt('partd_justin.csv' , delimiter=',', unpack=True)
        d_michael = np.loadtxt('partd_michael.csv', delimiter=',', unpack=True)
        
        n = 2048
        x_j, y_j = d_justin[0][:n], d_justin[1][:n]
        x_m, y_m = d_michael[0][:n], d_michael[1][:n]
        window = np.hanning(n)

        y_j = np.multiply(y_j, window)
        y_m = np.multiply(y_m, window)

        freqs_j, fft_j, ft_j = FFT(x_j, y_j)
        freqs_m, fft_m, ft_m = FFT(x_m, y_m)

        ft_j = [100*f/max(ft_j) for f in ft_j]
        ft_m = [100*f/max(ft_m) for f in ft_m]

        plt.plot(freqs_j, ft_j)
        plt.plot(freqs_m, ft_m)
        plt.xlim([1000, 3000])
        plt.show()


    def PartE():
        print("#"*40)
        print("Part E:")
        print("#"*10)

        def fft_from_file(file):
            n = 2048
            d_f  = np.loadtxt(file , delimiter=',', unpack=True)
            x_f, y_f = d_f[0][:n], d_f[1][:n]
            window = np.hanning(n)
            y_f = np.multiply(y_f, window)
            return FFT(x_f, y_f)

        people = 'michael', 'justin'
        vowels = 'a', 'e', 'i', 'o', 'u'
        for v in vowels:
            for p in people:
                freqs_p_v, fft_p_v, ft_p_v = fft_from_file('parte_'+p+'_'+v+'.csv')
                ft_p_v = [100*f/max(ft_p_v) for f in ft_p_v]
                plt.title('Vowel: '+v+'\nBlue: Michael, Orange: Justin')
                plt.plot(freqs_p_v, ft_p_v)
            plt.show()

    def PartF():
        print("#"*40)
        print("Part F:")
        print("#"*10)

        def find_peaks(x, y):
            xs = np.array([])
            peaks = np.array([])
            buf = 2
            for i, amp in enumerate(y):
                if i<buf or len(y)-i<buf: continue
                if max(y[i-buf:i+buf])==amp and amp>=5:
                    peaks = np.append(peaks, amp)
                    xs = np.append(xs, x[i])
            scale = max(peaks)/100.
            peaks = [yi/scale for yi in peaks if yi/scale>=5]
            f0 = xs[0]
            xs = np.divide(xs, f0)
            return xs, peaks

        d_voice  = np.loadtxt('partf_voice.csv' , delimiter=',', unpack=True, dtype=np.float64)
        d_guitar = np.loadtxt('partf_guitar.csv', delimiter=',', unpack=True, dtype=np.float64)
        d_piano = np.loadtxt('partf_piano.csv', delimiter=',', unpack=True, dtype=np.float64)
        
        n = 2048
        x_v, y_v = d_voice[0][:n], d_voice[1][:n]
        x_g, y_g = d_guitar[0][:n], d_guitar[1][:n]
        x_p, y_p = d_piano[0][:n], d_piano[1][:n]
        window = np.hanning(n)

        y_v = np.multiply(y_v, window)
        y_g = np.multiply(y_g, window)
        y_p = np.multiply(y_p, window)

        freqs_v, fft_v, ft_v = FFT(x_v, y_v)
        freqs_g, fft_g, ft_g = FFT(x_g, y_g)
        freqs_p, fft_p, ft_p = FFT(x_p, y_p)

        ft_v = [100*f/max(ft_v) for f in ft_v]
        ft_g = [100*f/max(ft_g) for f in ft_g]
        ft_p = [100*f/max(ft_p) for f in ft_p]

        plt.plot(freqs_v, ft_v)
        plt.plot(freqs_g, ft_g)
        plt.plot(freqs_p, ft_p)
        plt.xlim([0, 10000])
        plt.title("Blue: voice, Orange: guitar, Green: piano")
        plt.show()

        harmonics_v, peaks_v = find_peaks(freqs_v, ft_v)
        harmonics_g, peaks_g = find_peaks(freqs_g, ft_g)
        harmonics_p, peaks_p = find_peaks(freqs_p, ft_p)

        plt.plot(harmonics_v, peaks_v, 'o')
        plt.plot(harmonics_g, peaks_g, 'o')
        plt.plot(harmonics_p, peaks_p, 'o')
        plt.title("Blue: voice, Orange: guitar, Green: piano")
        plt.show()


    PartA()
    PartB()
    PartC()
    PartD()
    PartE()
    PartF()
    return

if __name__=="__main__":
    main()

