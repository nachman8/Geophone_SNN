import numpy as np

from sctn.resonator import simple_resonator

import numpy as np

from sctn.resonator import simple_resonator


###########################################
# Resonator functions from resonator.txt data
###########################################

def resonator_0_100():
    resonator = simple_resonator(
        freq0=0.100,
        clk_freq=1536,
        lf=4,
        thetas=[-3.449, -10.563, -9.532, -9.625],
        weights=[23.168, 16.655, 21.389, 19.051, 19.339],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_0_250():
    resonator = simple_resonator(
        freq0=0.250,
        clk_freq=1536,
        lf=4,
        thetas=[-30.365, -26.786, -24.465, -24.437],
        weights=[99.455, 40.538, 52.741, 48.947, 49.019],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_0_372():
    resonator = simple_resonator(
        freq0=0.372,
        clk_freq=1536,
        lf=4,
        thetas=[-48.663, -40.06, -35.864, -35.336],
        weights=[157.321, 62.183, 79.307, 71.374, 71.063],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_0_694():
    resonator = simple_resonator(
        freq0=0.694,
        clk_freq=1536,
        lf=4,
        thetas=[-113.694, -77.332, -69.207, -62.432],
        weights=[334.262, 112.405, 154.573, 135.118, 126.891],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_0_898():
    resonator = simple_resonator(
        freq0=0.898,
        clk_freq=1536,
        lf=4,
        thetas=[-106.516, -93.167, -90.835, -81.611],
        weights=[359.026, 152.001, 182.454, 179.91, 167.481],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_1_00():
    resonator = simple_resonator(
        freq0=1.00,
        clk_freq=15360,
        lf=4,
        thetas=[-3.449, -10.563, -9.532, -9.625],
        weights=[23.168, 16.655, 21.389, 19.051, 19.339],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_1_30():
    resonator = simple_resonator(
        freq0=1.30,
        clk_freq=15360,
        lf=4,
        thetas=[-38.705, -14.082, -11.844, -12.294],
        weights=[97.872, 21.307, 27.448, 24.055, 24.879],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_1_59():
    resonator = simple_resonator(
        freq0=1.59,
        clk_freq=15360,
        lf=4,
        thetas=[-18.833, -17.091, -14.972, -15.48],
        weights=[63.287, 26.537, 33.677, 30.133, 31.076],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_1_75():
    resonator = simple_resonator(
        freq0=1.75,
        clk_freq=15360,
        lf=4,
        thetas=[-24.442, -17.659, -17.053, -16.969],
        weights=[78.467, 29.833, 35.414, 34.069, 33.882],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_1_90():
    resonator = simple_resonator(
        freq0=1.90,
        clk_freq=15360,
        lf=4,
        thetas=[-22.972, -20.365, -18.31, -18.796],
        weights=[75.971, 31.307, 40.053, 36.719, 37.694],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_1_95():
    resonator = simple_resonator(
        freq0=1.95,
        clk_freq=15360,
        lf=4,
        thetas=[-23.292, -20.967, -18.739, -19.253],
        weights=[77.357, 32.09, 41.255, 37.596, 38.596],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_2_50():
    resonator = simple_resonator(
        freq0=2.50,
        clk_freq=15360,
        lf=4,
        thetas=[-30.365, -26.786, -24.465, -24.437],
        weights=[99.455, 40.538, 52.741, 48.947, 49.019],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_2_68():
    resonator = simple_resonator(
        freq0=2.68,
        clk_freq=15360,
        lf=4,
        thetas=[-8.927, -25.704, -25.206, -26.618],
        weights=[65.351, 47.423, 51.518, 50.371, 53.112],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_2_79():
    resonator = simple_resonator(
        freq0=2.79,
        clk_freq=15360,
        lf=4,
        thetas=[-87.121, -27.251, -27.245, -28.08],
        weights=[223.098, 48.757, 55.048, 54.062, 55.662],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_3_05():
    resonator = simple_resonator(
        freq0=3.05,
        clk_freq=15360,
        lf=4,
        thetas=[-37.124, -32.549, -29.814, -29.008],
        weights=[123.204, 50.851, 64.454, 59.25, 58.383],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_3_39():
    resonator = simple_resonator(
        freq0=3.39,
        clk_freq=15360,
        lf=4,
        thetas=[-45.112, -36.415, -32.509, -32.316],
        weights=[144.962, 56.675, 72.227, 64.768, 64.959],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_3_47():
    resonator = simple_resonator(
        freq0=3.47,
        clk_freq=15360,
        lf=4,
        thetas=[-44.809, -37.682, -34.249, -33.187],
        weights=[142.565, 55.948, 73.91, 68.354, 67.02],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_3_72():
    resonator = simple_resonator(
        freq0=3.72,
        clk_freq=15360,
        lf=4,
        thetas=[-48.663, -40.06, -35.864, -35.336],
        weights=[157.321, 62.183, 79.307, 71.374, 71.063],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_5_09():
    resonator = simple_resonator(
        freq0=5.09,
        clk_freq=15360,
        lf=4,
        thetas=[-69.695, -55.164, -50.732, -47.095],
        weights=[219.153, 83.76, 109.522, 99.543, 95.232],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_5_26():
    resonator = simple_resonator(
        freq0=5.26,
        clk_freq=15360,
        lf=4,
        thetas=[-78.345, -57.279, -52.931, -48.217],
        weights=[239.144, 86.311, 114.419, 103.404, 97.135],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_5_45():
    resonator = simple_resonator(
        freq0=5.45,
        clk_freq=15360,
        lf=4,
        thetas=[-81.139, -59.604, -54.623, -49.983],
        weights=[247.52, 89.124, 119.23, 106.752, 100.593],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_5_87():
    resonator = simple_resonator(
        freq0=5.87,
        clk_freq=15360,
        lf=4,
        thetas=[-88.619, -64.356, -59.071, -53.569],
        weights=[268.708, 95.611, 128.823, 115.302, 107.954],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_6_36():
    resonator = simple_resonator(
        freq0=6.36,
        clk_freq=15360,
        lf=4,
        thetas=[-98.054, -70.205, -64.087, -57.686],
        weights=[294.356, 103.068, 140.527, 124.915, 116.543],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_6_94():
    resonator = simple_resonator(
        freq0=6.94,
        clk_freq=15360,
        lf=4,
        thetas=[-113.694, -77.332, -69.207, -62.432],
        weights=[334.262, 112.405, 154.573, 135.118, 126.891],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_7_63():
    resonator = simple_resonator(
        freq0=7.63,
        clk_freq=15360,
        lf=4,
        thetas=[-127.835, -85.872, -76.422, -67.925],
        weights=[371.729, 122.996, 171.202, 148.778, 139.294],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_8_98():
    resonator = simple_resonator(
        freq0=8.98,
        clk_freq=15360,
        lf=4,
        thetas=[-106.516, -93.167, -90.835, -81.611],
        weights=[359.026, 152.001, 182.454, 179.91, 167.481],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_9_54():
    resonator = simple_resonator(
        freq0=9.54,
        clk_freq=15360,
        lf=4,
        thetas=[-117.549, -100.74, -96.393, -85.843],
        weights=[387.993, 160.185, 197.341, 190.822, 177.199],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_0():
    resonator = simple_resonator(
        freq0=10.051891142646022,
        clk_freq=153600,
        lf=4,
        thetas=[-3.449, -10.563, -9.532, -9.625],
        weights=[23.168, 16.655, 21.389, 19.051, 19.339],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_1():
    resonator = simple_resonator(
        freq0=10.118459958160233,
        clk_freq=153600,
        lf=4,
        thetas=[-3.677, -10.151, -9.348, -9.831],
        weights=[24.724, 17.573, 20.542, 18.76, 19.907],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_2():
    resonator = simple_resonator(
        freq0=10.254278212632183,
        clk_freq=153600,
        lf=4,
        thetas=[-3.742, -10.123, -9.642, -10.086],
        weights=[25.436, 18.289, 20.427, 19.201, 20.276],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_3():
    resonator = simple_resonator(
        freq0=10.323563876231049,
        clk_freq=153600,
        lf=4,
        thetas=[-3.512, -9.746, -9.3, -9.848],
        weights=[27.396, 20.27, 19.597, 18.477, 19.634],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_4():
    resonator = simple_resonator(
        freq0=10.464982559467092,
        clk_freq=153600,
        lf=4,
        thetas=[-3.512, -9.747, -9.382, -10.468],
        weights=[27.533, 20.43, 19.662, 18.7, 20.884],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_5():
    resonator = simple_resonator(
        freq0=10.537154852980658,
        clk_freq=153600,
        lf=4,
        thetas=[-3.512, -9.747, -9.425, -10.468],
        weights=[27.533, 20.43, 19.662, 18.786, 20.884],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_6():
    resonator = simple_resonator(
        freq0=10.61032953945969,
        clk_freq=153600,
        lf=4,
        thetas=[-3.154, -10.055, -9.933, -10.832],
        weights=[26.663, 19.644, 19.955, 19.63, 21.508],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_10_7():
    resonator = simple_resonator(
        freq0=10.75977080057884,
        clk_freq=153600,
        lf=4,
        thetas=[-3.155, -10.056, -10.046, -11.105],
        weights=[26.663, 19.644, 19.955, 19.854, 22.052],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_1():
    resonator = simple_resonator(
        freq0=11.15246316556347,
        clk_freq=153600,
        lf=4,
        thetas=[-2.784, -10.616, -11.382, -10.791],
        weights=[24.624, 19.288, 21.458, 22.507, 21.457],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_2():
    resonator = simple_resonator(
        freq0=11.23446657119261,
        clk_freq=153600,
        lf=4,
        thetas=[-2.817, -10.696, -11.42, -11.089],
        weights=[24.683, 19.356, 21.618, 22.582, 22.049],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_3():
    resonator = simple_resonator(
        freq0=11.317684842090335,
        clk_freq=153600,
        lf=4,
        thetas=[-2.955, -10.834, -11.508, -11.174],
        weights=[25.032, 19.468, 21.813, 22.767, 22.297],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_4():
    resonator = simple_resonator(
        freq0=11.4021451767328,
        clk_freq=153600,
        lf=4,
        thetas=[-3.014, -10.882, -11.508, -11.174],
        weights=[25.077, 19.75, 21.99, 22.758, 22.219],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_5():
    resonator = simple_resonator(
        freq0=11.574904952137842,
        clk_freq=153600,
        lf=4,
        thetas=[-3.124, -10.985, -11.596, -11.55],
        weights=[25.576, 19.712, 22.241, 23.111, 23.204],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_6():
    resonator = simple_resonator(
        freq0=11.663263005207597,
        clk_freq=153600,
        lf=4,
        thetas=[-3.254, -11.102, -11.707, -11.55],
        weights=[25.959, 19.837, 22.5, 23.398, 23.217],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_7():
    resonator = simple_resonator(
        freq0=11.752980412939964,
        clk_freq=153600,
        lf=4,
        thetas=[-3.383, -11.206, -11.707, -11.549],
        weights=[25.881, 20.491, 22.685, 23.227, 23.077],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_8():
    resonator = simple_resonator(
        freq0=11.844088788234073,
        clk_freq=153600,
        lf=4,
        thetas=[-3.589, -11.391, -11.916, -11.172],
        weights=[26.808, 20.117, 23.144, 23.907, 22.573],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_11_9():
    resonator = simple_resonator(
        freq0=11.93662073189215,
        clk_freq=153600,
        lf=4,
        thetas=[-3.757, -11.514, -11.916, -11.485],
        weights=[27.146, 20.455, 23.39, 23.907, 23.199],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_0():
    resonator = simple_resonator(
        freq0=12.030609871513349,
        clk_freq=153600,
        lf=4,
        thetas=[-3.756, -11.638, -11.916, -11.56],
        weights=[27.146, 20.455, 23.638, 23.907, 23.349],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_1():
    resonator = simple_resonator(
        freq0=12.126090902239644,
        clk_freq=153600,
        lf=4,
        thetas=[-3.755, -11.899, -12.135, -11.346],
        weights=[27.235, 20.392, 24.221, 24.395, 22.962],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_2():
    resonator = simple_resonator(
        freq0=12.223099629457563,
        clk_freq=153600,
        lf=4,
        thetas=[-4.915, -12.109, -11.999, -11.426],
        weights=[30.796, 21.087, 24.635, 23.913, 22.764],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_3():
    resonator = simple_resonator(
        freq0=12.321673013566091,
        clk_freq=153600,
        lf=4,
        thetas=[-4.911, -11.978, -11.732, -12.133],
        weights=[31.071, 21.35, 24.294, 23.329, 24.149],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_4():
    resonator = simple_resonator(
        freq0=12.421849216928416,
        clk_freq=153600,
        lf=4,
        thetas=[-5.103, -12.549, -12.64, -11.874],
        weights=[30.002, 20.334, 25.044, 25.222, 23.763],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_5():
    resonator = simple_resonator(
        freq0=12.523667653132748,
        clk_freq=153600,
        lf=4,
        thetas=[-5.212, -12.758, -12.64, -11.874],
        weights=[30.22, 20.552, 25.462, 25.222, 23.763],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_6():
    resonator = simple_resonator(
        freq0=12.627169038695829,
        clk_freq=153600,
        lf=4,
        thetas=[-5.521, -12.896, -12.638, -11.803],
        weights=[31.288, 20.653, 26.059, 25.539, 23.827],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_7():
    resonator = simple_resonator(
        freq0=12.732395447351628,
        clk_freq=153600,
        lf=4,
        thetas=[-5.339, -12.758, -12.64, -12.118],
        weights=[30.66, 20.735, 25.721, 25.479, 24.489],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_8():
    resonator = simple_resonator(
        freq0=12.839390367077272,
        clk_freq=153600,
        lf=4,
        thetas=[-5.336, -12.763, -12.648, -12.412],
        weights=[31.704, 21.255, 25.791, 25.167, 24.932],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_12_9():
    resonator = simple_resonator(
        freq0=12.948198760018604,
        clk_freq=153600,
        lf=4,
        thetas=[-5.336, -12.885, -12.742, -12.589],
        weights=[31.794, 21.257, 26.104, 25.419, 25.316],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_0():
    resonator = simple_resonator(
        freq0=13.058867125488849,
        clk_freq=153600,
        lf=4,
        thetas=[-5.729, -13.147, -12.849, -12.734],
        weights=[32.904, 21.713, 26.682, 25.69, 25.602],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_1():
    resonator = simple_resonator(
        freq0=13.17144356622582,
        clk_freq=153600,
        lf=4,
        thetas=[-5.518, -13.017, -12.978, -12.929],
        weights=[32.941, 22.035, 26.231, 25.667, 25.731],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_2():
    resonator = simple_resonator(
        freq0=13.285977858106046,
        clk_freq=153600,
        lf=4,
        thetas=[-5.674, -13.018, -13.075, -13.097],
        weights=[33.279, 22.408, 26.234, 25.909, 26.131],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_4():
    resonator = simple_resonator(
        freq0=13.40252152352803,
        clk_freq=153600,
        lf=4,
        thetas=[-5.852, -13.168, -13.369, -13.346],
        weights=[33.902, 22.479, 26.528, 26.629, 26.716],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_5():
    resonator = simple_resonator(
        freq0=13.521127908691994,
        clk_freq=153600,
        lf=4,
        thetas=[-5.834, -13.149, -13.348, -13.425],
        weights=[33.67, 22.755, 26.556, 26.556, 26.882],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_6():
    resonator = simple_resonator(
        freq0=13.641852265019601,
        clk_freq=153600,
        lf=4,
        thetas=[-6.051, -13.492, -13.511, -13.566],
        weights=[34.496, 22.671, 27.3, 27.009, 27.182],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_7():
    resonator = simple_resonator(
        freq0=13.764751834974732,
        clk_freq=153600,
        lf=4,
        thetas=[-5.832, -13.458, -13.354, -13.726],
        weights=[34.663, 23.157, 27.06, 26.466, 27.41],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_13_8():
    resonator = simple_resonator(
        freq0=13.88988594256541,
        clk_freq=153600,
        lf=4,
        thetas=[-5.971, -13.571, -13.449, -13.889],
        weights=[35.002, 23.503, 27.353, 26.758, 27.821],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_0():
    resonator = simple_resonator(
        freq0=14.017316088827481,
        clk_freq=153600,
        lf=4,
        thetas=[-6.16, -13.736, -13.784, -14.142],
        weights=[35.702, 23.517, 27.673, 27.479, 28.321],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_1():
    resonator = simple_resonator(
        freq0=14.14710605261292,
        clk_freq=153600,
        lf=4,
        thetas=[-6.166, -13.546, -13.828, -14.212],
        weights=[35.685, 23.729, 27.63, 27.394, 28.205],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_2():
    resonator = simple_resonator(
        freq0=14.279321997029863,
        clk_freq=153600,
        lf=4,
        thetas=[-6.217, -13.799, -13.853, -14.637],
        weights=[35.756, 23.441, 27.693, 27.632, 29.271],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_4():
    resonator = simple_resonator(
        freq0=14.414032581907502,
        clk_freq=153600,
        lf=4,
        thetas=[-6.217, -13.969, -13.999, -14.807],
        weights=[35.84, 23.499, 28.14, 28.014, 29.661],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_5():
    resonator = simple_resonator(
        freq0=14.551309082687574,
        clk_freq=153600,
        lf=4,
        thetas=[-6.217, -14.127, -14.131, -14.948],
        weights=[35.935, 23.532, 28.552, 28.343, 29.935],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_6():
    resonator = simple_resonator(
        freq0=14.691225516174953,
        clk_freq=153600,
        lf=4,
        thetas=[-6.393, -14.133, -14.275, -15.118],
        weights=[36.927, 24.541, 28.597, 28.29, 29.958],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_7():
    resonator = simple_resonator(
        freq0=14.833858773613548,
        clk_freq=153600,
        lf=4,
        thetas=[-6.137, -14.029, -14.494, -15.015],
        weights=[37.951, 25.58, 28.259, 28.687, 29.889],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_8():
    resonator = simple_resonator(
        freq0=14.833858773613548,
        clk_freq=153600,
        lf=4,
        thetas=[-6.22, -14.024, -14.587, -15.053],
        weights=[37.074, 24.515, 28.447, 29.182, 30.1],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_14_9():
    resonator = simple_resonator(
        freq0=14.97928876159015,
        clk_freq=153600,
        lf=4,
        thetas=[-6.44, -14.186, -14.797, -15.521],
        weights=[37.128, 24.599, 28.556, 29.357, 30.908],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_15_1():
    resonator = simple_resonator(
        freq0=15.127598551308864,
        clk_freq=153600,
        lf=4,
        thetas=[-6.44, -14.375, -14.949, -15.689],
        weights=[37.199, 24.648, 29.019, 29.769, 31.354],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_15_2():
    resonator = simple_resonator(
        freq0=15.278874536821952,
        clk_freq=153600,
        lf=4,
        thetas=[-6.44, -14.567, -15.101, -15.866],
        weights=[37.265, 24.669, 29.473, 30.137, 31.749],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_15_4():
    resonator = simple_resonator(
        freq0=15.433206602850456,
        clk_freq=153600,
        lf=4,
        thetas=[-6.44, -14.78, -15.27, -16.055],
        weights=[37.299, 24.676, 29.933, 30.503, 32.141],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_15_5():
    resonator = simple_resonator(
        freq0=15.590688302879546,
        clk_freq=153600,
        lf=4,
        thetas=[-7.359, -15.193, -15.615, -16.649],
        weights=[39.232, 24.705, 30.378, 31.176, 33.208],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_15_6():
    resonator = simple_resonator(
        freq0=15.751417048270055,
        clk_freq=153600,
        lf=4,
        thetas=[-7.373, -15.144, -15.505, -15.899],
        weights=[40.369, 25.843, 30.651, 30.867, 31.744],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_15_7():
    resonator = simple_resonator(
        freq0=15.751417048270055,
        clk_freq=153600,
        lf=4,
        thetas=[-7.437, -15.228, -15.677, -15.813],
        weights=[40.229, 25.593, 30.796, 31.192, 31.567],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_15_9():
    resonator = simple_resonator(
        freq0=15.915494309189533,
        clk_freq=153600,
        lf=4,
        thetas=[-7.704, -15.43, -15.796, -16.745],
        weights=[40.072, 24.908, 30.894, 31.573, 33.425],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_16_0():
    resonator = simple_resonator(
        freq0=16.083025828233634,
        clk_freq=153600,
        lf=4,
        thetas=[-7.834, -15.606, -15.937, -16.792],
        weights=[40.952, 25.272, 31.537, 31.844, 33.296],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_16_2():
    resonator = simple_resonator(
        freq0=16.254121847682928,
        clk_freq=153600,
        lf=4,
        thetas=[-7.841, -15.613, -16.111, -16.791],
        weights=[41.398, 26.159, 31.637, 31.961, 33.341],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_16_3():
    resonator = simple_resonator(
        freq0=16.428897351421455,
        clk_freq=153600,
        lf=4,
        thetas=[-7.84, -15.628, -16.264, -16.938],
        weights=[42.317, 26.795, 31.495, 32.302, 33.782],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_16_4():
    resonator = simple_resonator(
        freq0=16.428897351421455,
        clk_freq=153600,
        lf=4,
        thetas=[-7.84, -15.63, -16.266, -16.939],
        weights=[42.433, 26.825, 31.455, 32.333, 33.761],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_16_6():
    resonator = simple_resonator(
        freq0=16.607472322632557,
        clk_freq=153600,
        lf=4,
        thetas=[-7.84, -15.746, -16.414, -17.085],
        weights=[43.017, 27.487, 31.616, 32.608, 34.029],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_16_7():
    resonator = simple_resonator(
        freq0=16.789972018485663,
        clk_freq=153600,
        lf=4,
        thetas=[-8.041, -16.119, -16.712, -17.311],
        weights=[43.739, 27.841, 32.505, 33.351, 34.571],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_16_9():
    resonator = simple_resonator(
        freq0=16.9765272631355,
        clk_freq=153600,
        lf=4,
        thetas=[-7.992, -16.039, -16.714, -17.401],
        weights=[44.011, 28.368, 32.333, 33.157, 34.593],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_17_0():
    resonator = simple_resonator(
        freq0=17.167274760474104,
        clk_freq=153600,
        lf=4,
        thetas=[-7.992, -16.159, -16.877, -17.57],
        weights=[44.586, 28.736, 32.551, 33.553, 35.008],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_17_1():
    resonator = simple_resonator(
        freq0=17.167274760474104,
        clk_freq=153600,
        lf=4,
        thetas=[-7.992, -16.159, -16.877, -17.57],
        weights=[44.598, 28.737, 32.548, 33.554, 35.014],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_17_3():
    resonator = simple_resonator(
        freq0=17.362357428206767,
        clk_freq=153600,
        lf=4,
        thetas=[-8.206, -16.51, -17.226, -17.867],
        weights=[45.692, 29.671, 33.135, 34.18, 35.593],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_17_4():
    resonator = simple_resonator(
        freq0=17.561924754967762,
        clk_freq=153600,
        lf=4,
        thetas=[-7.755, -16.206, -17.104, -17.593],
        weights=[46.467, 31.119, 32.549, 33.876, 35.101],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_17_5():
    resonator = simple_resonator(
        freq0=17.561924754967762,
        clk_freq=153600,
        lf=4,
        thetas=[-8.174, -16.739, -17.183, -17.634],
        weights=[46.134, 29.907, 33.567, 34.079, 35.198],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_17_6():
    resonator = simple_resonator(
        freq0=17.76613318235111,
        clk_freq=153600,
        lf=4,
        thetas=[-8.682, -17.339, -17.833, -17.968],
        weights=[45.0, 28.109, 35.193, 35.525, 35.763],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_17_9():
    resonator = simple_resonator(
        freq0=17.97514651390818,
        clk_freq=153600,
        lf=4,
        thetas=[-8.974, -17.393, -17.77, -18.632],
        weights=[46.254, 28.584, 34.858, 35.554, 37.23],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_1():
    resonator = simple_resonator(
        freq0=18.189136353359466,
        clk_freq=153600,
        lf=4,
        thetas=[-8.973, -17.548, -17.978, -18.641],
        weights=[47.219, 29.472, 35.363, 35.749, 37.139],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_2():
    resonator = simple_resonator(
        freq0=18.40828257448428,
        clk_freq=153600,
        lf=4,
        thetas=[-8.972, -17.587, -18.02, -18.659],
        weights=[48.693, 30.852, 35.322, 35.851, 37.192],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_3():
    resonator = simple_resonator(
        freq0=18.40828257448428,
        clk_freq=153600,
        lf=4,
        thetas=[-8.972, -17.587, -18.021, -18.659],
        weights=[48.703, 30.86, 35.324, 35.86, 37.195],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_4():
    resonator = simple_resonator(
        freq0=18.40828257448428,
        clk_freq=153600,
        lf=4,
        thetas=[-8.972, -17.587, -18.021, -18.659],
        weights=[48.709, 30.862, 35.323, 35.86, 37.193],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_5():
    resonator = simple_resonator(
        freq0=18.632773825392626,
        clk_freq=153600,
        lf=4,
        thetas=[-8.972, -17.616, -18.192, -18.822],
        weights=[49.581, 31.698, 35.386, 36.194, 37.558],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_6():
    resonator = simple_resonator(
        freq0=18.632773825392626,
        clk_freq=153600,
        lf=4,
        thetas=[-8.972, -17.611, -18.187, -18.819],
        weights=[49.656, 31.772, 35.383, 36.183, 37.51],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_8():
    resonator = simple_resonator(
        freq0=18.86280807015056,
        clk_freq=153600,
        lf=4,
        thetas=[-9.171, -17.86, -18.554, -19.132],
        weights=[50.957, 32.997, 35.851, 36.909, 38.158],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_18_9():
    resonator = simple_resonator(
        freq0=19.098593171027442,
        clk_freq=153600,
        lf=4,
        thetas=[-8.972, -17.667, -18.538, -19.149],
        weights=[51.552, 33.657, 35.511, 36.892, 38.165],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_0():
    resonator = simple_resonator(
        freq0=19.098593171027442,
        clk_freq=153600,
        lf=4,
        thetas=[-8.972, -17.669, -18.541, -19.151],
        weights=[51.57, 33.699, 35.473, 36.907, 38.19],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_1():
    resonator = simple_resonator(
        freq0=19.340347514964495,
        clk_freq=153600,
        lf=4,
        thetas=[-8.659, -17.771, -18.604, -19.4],
        weights=[51.878, 34.603, 35.712, 36.888, 38.741],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_2():
    resonator = simple_resonator(
        freq0=19.340347514964495,
        clk_freq=153600,
        lf=4,
        thetas=[-8.653, -17.76, -18.598, -19.427],
        weights=[51.86, 34.551, 35.711, 36.941, 38.765],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_3():
    resonator = simple_resonator(
        freq0=19.340347514964495,
        clk_freq=153600,
        lf=4,
        thetas=[-8.611, -17.825, -18.558, -19.235],
        weights=[51.979, 34.763, 35.839, 36.885, 38.395],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_4():
    resonator = simple_resonator(
        freq0=19.588300688233275,
        clk_freq=153600,
        lf=4,
        thetas=[-9.47, -18.611, -19.13, -20.124],
        weights=[52.582, 33.72, 37.35, 38.228, 40.054],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_5():
    resonator = simple_resonator(
        freq0=19.588300688233275,
        clk_freq=153600,
        lf=4,
        thetas=[-9.469, -18.608, -19.129, -20.123],
        weights=[52.595, 33.734, 37.345, 38.188, 40.06],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_6():
    resonator = simple_resonator(
        freq0=19.842694203664873,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.323, -19.109, -20.156],
        weights=[53.312, 34.775, 36.844, 38.003, 40.199],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_8():
    resonator = simple_resonator(
        freq0=19.842694203664873,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.324, -19.109, -20.156],
        weights=[53.333, 34.798, 36.846, 38.017, 40.19],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_19_9():
    resonator = simple_resonator(
        freq0=20.103782285292045,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.531, -19.3, -20.336],
        weights=[54.123, 35.546, 37.251, 38.434, 40.519],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_1():
    resonator = simple_resonator(
        freq0=20.103782285292045,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.53, -19.301, -20.336],
        weights=[54.158, 35.585, 37.265, 38.424, 40.526],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_2():
    resonator = simple_resonator(
        freq0=20.371832715762604,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.558, -19.688, -20.533],
        weights=[54.868, 36.442, 37.319, 39.144, 40.95],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_3():
    resonator = simple_resonator(
        freq0=20.371832715762604,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.558, -19.688, -20.534],
        weights=[54.881, 36.44, 37.317, 39.164, 40.955],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_4():
    resonator = simple_resonator(
        freq0=20.647127752462097,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.703, -20.105, -20.723],
        weights=[55.464, 37.063, 37.608, 39.948, 41.322],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_6():
    resonator = simple_resonator(
        freq0=20.647127752462097,
        clk_freq=153600,
        lf=4,
        thetas=[-9.273, -18.699, -20.101, -20.722],
        weights=[55.521, 37.064, 37.609, 39.966, 41.339],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_7():
    resonator = simple_resonator(
        freq0=20.929965118934184,
        clk_freq=153600,
        lf=4,
        thetas=[-12.069, -20.925, -21.69, -21.749],
        weights=[55.838, 31.985, 41.889, 43.298, 43.413],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_8():
    resonator = simple_resonator(
        freq0=20.929965118934184,
        clk_freq=153600,
        lf=4,
        thetas=[-12.107, -20.923, -21.683, -21.746],
        weights=[56.027, 32.109, 41.837, 43.267, 43.365],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_20_9():
    resonator = simple_resonator(
        freq0=20.929965118934184,
        clk_freq=153600,
        lf=4,
        thetas=[-12.141, -20.924, -21.688, -21.748],
        weights=[56.103, 32.152, 41.835, 43.253, 43.336],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_21_3():
    resonator = simple_resonator(
        freq0=21.51954160115768,
        clk_freq=153600,
        lf=4,
        thetas=[-11.005, -20.445, -20.878, -20.878],
        weights=[60.142, 38.155, 40.916, 41.633, 41.675],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_21_4():
    resonator = simple_resonator(
        freq0=21.51954160115768,
        clk_freq=153600,
        lf=4,
        thetas=[-11.005, -20.447, -20.882, -20.88],
        weights=[60.139, 38.2, 40.913, 41.649, 41.676],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_21_5():
    resonator = simple_resonator(
        freq0=21.51954160115768,
        clk_freq=153600,
        lf=4,
        thetas=[-11.005, -20.446, -20.881, -20.88],
        weights=[60.143, 38.202, 40.911, 41.641, 41.688],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_21_6():
    resonator = simple_resonator(
        freq0=21.82696362403136,
        clk_freq=153600,
        lf=4,
        thetas=[-10.508, -20.528, -20.99, -21.067],
        weights=[60.701, 39.674, 41.111, 41.894, 42.064],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_21_7():
    resonator = simple_resonator(
        freq0=21.82696362403136,
        clk_freq=153600,
        lf=4,
        thetas=[-10.36, -20.536, -21.001, -21.072],
        weights=[60.376, 39.662, 41.094, 41.878, 42.091],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_21_8():
    resonator = simple_resonator(
        freq0=21.82696362403136,
        clk_freq=153600,
        lf=4,
        thetas=[-10.36, -20.536, -21.002, -21.072],
        weights=[60.39, 39.674, 41.117, 41.89, 42.097],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_21_9():
    resonator = simple_resonator(
        freq0=22.143296430176743,
        clk_freq=153600,
        lf=4,
        thetas=[-10.36, -20.589, -21.07, -21.408],
        weights=[61.672, 40.923, 41.34, 41.965, 42.738],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_1():
    resonator = simple_resonator(
        freq0=22.143296430176743,
        clk_freq=153600,
        lf=4,
        thetas=[-10.36, -20.594, -21.077, -21.411],
        weights=[61.731, 40.977, 41.292, 42.022, 42.754],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_2():
    resonator = simple_resonator(
        freq0=22.46893314238522,
        clk_freq=153600,
        lf=4,
        thetas=[-10.36, -20.659, -21.159, -21.719],
        weights=[63.269, 42.453, 41.447, 42.178, 43.332],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_3():
    resonator = simple_resonator(
        freq0=22.46893314238522,
        clk_freq=153600,
        lf=4,
        thetas=[-10.196, -20.67, -21.175, -21.727],
        weights=[62.919, 42.452, 41.436, 42.242, 43.346],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_4():
    resonator = simple_resonator(
        freq0=22.46893314238522,
        clk_freq=153600,
        lf=4,
        thetas=[-10.196, -20.671, -21.177, -21.728],
        weights=[62.929, 42.448, 41.419, 42.244, 43.355],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_5():
    resonator = simple_resonator(
        freq0=22.8042903534656,
        clk_freq=153600,
        lf=4,
        thetas=[-9.847, -20.788, -21.329, -22.076],
        weights=[63.534, 43.808, 41.644, 42.516, 44.114],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_6():
    resonator = simple_resonator(
        freq0=22.8042903534656,
        clk_freq=153600,
        lf=4,
        thetas=[-9.848, -20.762, -21.295, -22.06],
        weights=[63.725, 43.929, 41.64, 42.483, 44.026],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_7():
    resonator = simple_resonator(
        freq0=22.8042903534656,
        clk_freq=153600,
        lf=4,
        thetas=[-9.69, -20.77, -21.306, -22.066],
        weights=[63.364, 43.873, 41.631, 42.52, 44.046],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_22_8():
    resonator = simple_resonator(
        freq0=22.8042903534656,
        clk_freq=153600,
        lf=4,
        thetas=[-9.69, -20.773, -21.31, -22.068],
        weights=[63.374, 43.872, 41.626, 42.505, 44.055],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_1():
    resonator = simple_resonator(
        freq0=23.149809904275685,
        clk_freq=153600,
        lf=4,
        thetas=[-9.69, -20.821, -21.519, -22.434],
        weights=[64.708, 45.187, 41.822, 42.86, 44.787],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_2():
    resonator = simple_resonator(
        freq0=23.505960825879928,
        clk_freq=153600,
        lf=4,
        thetas=[-9.69, -20.886, -21.742, -22.793],
        weights=[66.144, 46.629, 41.98, 43.307, 45.488],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_3():
    resonator = simple_resonator(
        freq0=23.505960825879928,
        clk_freq=153600,
        lf=4,
        thetas=[-9.69, -20.887, -21.743, -22.794],
        weights=[66.161, 46.61, 41.969, 43.302, 45.511],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_4():
    resonator = simple_resonator(
        freq0=23.505960825879928,
        clk_freq=153600,
        lf=4,
        thetas=[-9.521, -20.925, -21.789, -22.816],
        weights=[65.699, 46.481, 41.943, 43.39, 45.522],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_5():
    resonator = simple_resonator(
        freq0=23.505960825879928,
        clk_freq=153600,
        lf=4,
        thetas=[-9.52, -20.901, -21.761, -22.803],
        weights=[65.748, 46.532, 41.95, 43.383, 45.524],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_6():
    resonator = simple_resonator(
        freq0=23.8732414637843,
        clk_freq=153600,
        lf=4,
        thetas=[-9.349, -20.968, -22.187, -23.333],
        weights=[66.619, 47.661, 42.009, 44.096, 46.536],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_7():
    resonator = simple_resonator(
        freq0=23.8732414637843,
        clk_freq=153600,
        lf=4,
        thetas=[-9.194, -20.975, -22.266, -23.337],
        weights=[66.253, 47.415, 41.986, 44.286, 46.541],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_8():
    resonator = simple_resonator(
        freq0=23.8732414637843,
        clk_freq=153600,
        lf=4,
        thetas=[-9.194, -20.978, -22.269, -23.338],
        weights=[66.15, 47.56, 41.944, 44.293, 46.557],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_23_9():
    resonator = simple_resonator(
        freq0=24.25218180447929,
        clk_freq=153600,
        lf=4,
        thetas=[-9.194, -20.985, -22.517, -23.724],
        weights=[66.404, 47.778, 42.481, 45.238, 47.593],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_24_0():
    resonator = simple_resonator(
        freq0=24.25218180447929,
        clk_freq=153600,
        lf=4,
        thetas=[-9.195, -21.063, -22.836, -24.107],
        weights=[66.493, 47.967, 42.28, 45.153, 47.823],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_24_2():
    resonator = simple_resonator(
        freq0=24.25218180447929,
        clk_freq=153600,
        lf=4,
        thetas=[-9.195, -21.004, -22.821, -24.142],
        weights=[66.532, 48.085, 42.091, 45.205, 47.919],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_24_5():
    resonator = simple_resonator(
        freq0=24.643346027132182,
        clk_freq=153600,
        lf=4,
        thetas=[-13.536, -23.638, -24.035, -23.792],
        weights=[69.933, 43.059, 47.247, 47.991, 47.619],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_24_6():
    resonator = simple_resonator(
        freq0=24.643346027132182,
        clk_freq=153600,
        lf=4,
        thetas=[-12.837, -23.626, -23.984, -23.843],
        weights=[68.883, 43.309, 47.122, 47.821, 47.769],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_24_7():
    resonator = simple_resonator(
        freq0=25.047335306265495,
        clk_freq=153600,
        lf=4,
        thetas=[-12.939, -23.554, -24.388, -24.567],
        weights=[70.248, 44.357, 47.08, 48.593, 49.006],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_24_8():
    resonator = simple_resonator(
        freq0=25.047335306265495,
        clk_freq=153600,
        lf=4,
        thetas=[-12.937, -23.537, -24.364, -24.557],
        weights=[70.214, 44.394, 47.078, 48.553, 49.018],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_0():
    resonator = simple_resonator(
        freq0=25.047335306265495,
        clk_freq=153600,
        lf=4,
        thetas=[-12.937, -23.531, -24.356, -24.554],
        weights=[70.194, 44.39, 47.087, 48.558, 49.006],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_1():
    resonator = simple_resonator(
        freq0=25.464790894703256,
        clk_freq=153600,
        lf=4,
        thetas=[-12.415, -23.705, -24.58, -24.658],
        weights=[71.345, 46.455, 47.424, 48.943, 49.209],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_2():
    resonator = simple_resonator(
        freq0=25.464790894703256,
        clk_freq=153600,
        lf=4,
        thetas=[-12.414, -23.688, -24.558, -24.648],
        weights=[71.348, 46.478, 47.442, 48.959, 49.195],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_3():
    resonator = simple_resonator(
        freq0=25.464790894703256,
        clk_freq=153600,
        lf=4,
        thetas=[-12.414, -23.691, -24.562, -24.649],
        weights=[71.349, 46.496, 47.437, 48.978, 49.191],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_4():
    resonator = simple_resonator(
        freq0=25.464790894703256,
        clk_freq=153600,
        lf=4,
        thetas=[-12.414, -23.692, -24.563, -24.649],
        weights=[71.346, 46.514, 47.435, 48.961, 49.21],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_5():
    resonator = simple_resonator(
        freq0=25.89639752003721,
        clk_freq=153600,
        lf=4,
        thetas=[-12.241, -23.811, -24.715, -24.986],
        weights=[72.974, 48.417, 47.71, 49.256, 49.907],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_6():
    resonator = simple_resonator(
        freq0=25.89639752003721,
        clk_freq=153600,
        lf=4,
        thetas=[-12.083, -23.822, -24.732, -24.994],
        weights=[72.625, 48.393, 47.731, 49.308, 49.89],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_7():
    resonator = simple_resonator(
        freq0=25.89639752003721,
        clk_freq=153600,
        lf=4,
        thetas=[-12.083, -23.825, -24.735, -24.996],
        weights=[72.63, 48.386, 47.738, 49.312, 49.875],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_8():
    resonator = simple_resonator(
        freq0=25.89639752003721,
        clk_freq=153600,
        lf=4,
        thetas=[-12.083, -23.824, -24.734, -24.996],
        weights=[72.631, 48.403, 47.733, 49.313, 49.877],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_25_9():
    resonator = simple_resonator(
        freq0=26.34288713245164,
        clk_freq=153600,
        lf=4,
        thetas=[-11.443, -23.999, -24.964, -25.393],
        weights=[73.215, 50.217, 48.058, 49.798, 50.699],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_26_0():
    resonator = simple_resonator(
        freq0=26.34288713245164,
        clk_freq=153600,
        lf=4,
        thetas=[-11.443, -23.999, -24.965, -25.393],
        weights=[73.213, 50.223, 48.062, 49.805, 50.699],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_26_1():
    resonator = simple_resonator(
        freq0=26.34288713245164,
        clk_freq=153600,
        lf=4,
        thetas=[-11.443, -24.0, -24.967, -25.394],
        weights=[73.217, 50.222, 48.052, 49.806, 50.7],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_26_3():
    resonator = simple_resonator(
        freq0=26.34288713245164,
        clk_freq=153600,
        lf=4,
        thetas=[-11.443, -24.001, -24.969, -25.395],
        weights=[73.209, 50.241, 48.065, 49.806, 50.712],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_26_4():
    resonator = simple_resonator(
        freq0=26.80504304705606,
        clk_freq=153600,
        lf=4,
        thetas=[-17.694, -27.104, -29.592, -27.323],
        weights=[73.693, 39.294, 53.889, 58.929, 54.605],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_26_8():
    resonator = simple_resonator(
        freq0=26.8,
        clk_freq=153600,
        lf=4,
        thetas=[-8.927, -25.704, -25.206, -26.618],
        weights=[65.351, 47.423, 51.518, 50.371, 53.112],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_27_9():
    resonator = simple_resonator(
        freq0=27.9,
        clk_freq=153600,
        lf=4,
        thetas=[-87.121, -27.251, -27.245, -28.08],
        weights=[223.098, 48.757, 55.048, 54.062, 55.662],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_28_8():
    resonator = simple_resonator(
        freq0=28.8,
        clk_freq=153600,
        lf=4,
        thetas=[-38.576, -29.943, -28.106, -27.831],
        weights=[123.81, 48.212, 59.177, 56.132, 55.85],
    )
    resonator.log_out_spikes(-1)
    return resonator




# resonator_functions.py
def resonator_31_3():
    resonator = simple_resonator(
        freq0=31.30916913283187,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.109, -3.155, -2.941, -2.698],
        weights=[7.233, 4.99, 6.324, 5.835, 5.423],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_31_8():
    resonator = simple_resonator(
        freq0=31.83098861837907,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.109, -3.273, -3.243, -2.96],
        weights=[7.086, 5.057, 6.426, 6.228, 5.818],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_32_1():
    resonator = simple_resonator(
        freq0=32.16605165646727,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.777, -3.133, -3.625, -2.939],
        weights=[7.241, 5.639, 6.187, 7.09, 5.878],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_32_5():
    resonator = simple_resonator(
        freq0=32.508243695365856,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.767, -3.206, -3.629, -3.076],
        weights=[7.29, 5.688, 6.327, 7.087, 6.128],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_32_8():
    resonator = simple_resonator(
        freq0=32.85779470284291,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.763, -2.999, -3.628, -3.706],
        weights=[6.468, 5.062, 5.927, 7.19, 7.339],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_33_4():
    resonator = simple_resonator(
        freq0=33.432985857378455,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.757, -3.351, -3.284, -3.566],
        weights=[7.431, 5.784, 6.618, 6.494, 7.061],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_33_7():
    resonator = simple_resonator(
        freq0=33.72819986053411,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.752, -3.224, -3.143, -3.511],
        weights=[7.199, 5.736, 6.216, 6.036, 6.675],
    )
    resonator.log_out_spikes(-1)
    return resonator







def resonator_34():
    resonator = simple_resonator(
        freq0=34.028673801385196,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.75, -3.226, -3.456, -3.477],
        weights=[7.438, 5.906, 6.379, 6.812, 6.902],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_35_2():
    resonator = simple_resonator(
        freq0=35.20478003876026,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.804, -3.748, -3.471, -3.394],
        weights=[7.261, 5.847, 7.522, 7.095, 6.879],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_35_6():
    resonator = simple_resonator(
        freq0=35.61509216042413,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.75, -3.752, -3.475, -3.396],
        weights=[7.464, 6.103, 7.461, 6.96, 6.879],
    )
    resonator.log_out_spikes(-1)
    return resonator



def resonator_36_9():
    resonator = simple_resonator(
        freq0=36.90549405029457,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.98, -3.752, -3.47, -3.382],
        weights=[8.859, 6.879, 7.44, 6.891, 6.788],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_36():
    resonator = simple_resonator(
        freq0=36.03508145476876,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.981, -3.75, -3.472, -3.39],
        weights=[8.302, 6.236, 7.407, 6.839, 6.813],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_37_3():
    resonator = simple_resonator(
        freq0=37.35666145922237,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.982, -3.753, -3.927, -3.842],
        weights=[7.57, 5.687, 7.514, 7.903, 7.759],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_37_8():
    resonator = simple_resonator(
        freq0=37.81899637827216,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.063, -3.489, -3.71, -3.647],
        weights=[9.56, 7.251, 6.923, 7.3, 7.124],
    )
    resonator.log_out_spikes(-1)
    return resonator



def resonator_38_4():
    resonator = simple_resonator(
        freq0=38.485830067561594,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.063, -3.488, -3.71, -3.647],
        weights=[9.56, 7.251, 6.923, 7.3, 7.124],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_38_6():
    resonator = simple_resonator(
        freq0=38.680695029929,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.063, -3.488, -3.708, -3.641],
        weights=[9.646, 7.137, 7.089, 7.595, 7.383],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_38_8():
    resonator = simple_resonator(
        freq0=38.877543350691994,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.229, -3.497, -3.705, -3.635],
        weights=[10.026, 7.144, 7.09, 7.567, 7.304],
    )
    resonator.log_out_spikes(-1)
    return resonator


def resonator_39_6():
    resonator = simple_resonator(
        freq0=39.68538840732975,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.403, -3.696, -3.702, -3.629],
        weights=[10.515, 7.147, 7.475, 7.513, 7.32],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_39_7():
    resonator = simple_resonator(
        freq0=39.78873577297384,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.551, -3.834, -3.701, -3.627],
        weights=[10.811, 7.443, 7.751, 7.513, 7.32],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_40_0():
    resonator = simple_resonator(
        freq0=40.102032905044496,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.551, -3.834, -3.701, -3.626],
        weights=[10.811, 7.443, 7.751, 7.513, 7.32],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_40_5():
    resonator = simple_resonator(
        freq0=40.52751866531022,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.551, -3.964, -3.701, -3.625],
        weights=[10.811, 7.443, 8.011, 7.513, 7.32],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_41_2():
    resonator = simple_resonator(
        freq0=41.2942555049242,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.413, -4.062, -4.214, -4.046],
        weights=[9.51, 6.737, 8.151, 8.476, 8.088],
    )
    resonator.log_out_spikes(-1)
    return resonator




def resonator_42_3():
    resonator = simple_resonator(
        freq0=42.3237521795622,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.005, -3.709, -3.684, -3.995],
        weights=[11.244, 8.703, 7.474, 7.409, 7.849],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_42_5():
    resonator = simple_resonator(
        freq0=42.559539099782604,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.385, -3.978, -3.825, -3.977],
        weights=[11.593, 8.411, 7.903, 7.732, 8.132],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_43_0():
    resonator = simple_resonator(
        freq0=43.03908320231536,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.385, -3.978, -3.825, -3.976],
        weights=[11.593, 8.411, 7.903, 7.732, 8.132],
    )
    resonator.log_out_spikes(-1)
    return resonator



def resonator_44_5():
    resonator = simple_resonator(
        freq0=44.54482372251298,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.76, -4.751, -3.815, -3.967],
        weights=[12.076, 8.349, 9.287, 7.63, 8.195],
    )
    resonator.log_out_spikes(-1)
    return resonator



def resonator_45_2():
    resonator = simple_resonator(
        freq0=45.20377081899986,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.77, -5.109, -4.252, -4.208],
        weights=[10.213, 6.712, 10.374, 8.738, 8.727],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_45_5():
    resonator = simple_resonator(
        freq0=45.60858070693121,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.337, -5.133, -4.54, -4.398],
        weights=[9.306, 7.093, 10.326, 9.132, 8.77],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_45_8():
    resonator = simple_resonator(
        freq0=45.88250611658244,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.782, -4.138, -4.232, -3.802],
        weights=[12.122, 9.969, 8.461, 8.741, 7.801],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_45_9():
    resonator = simple_resonator(
        freq0=46.020706436210695,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.782, -3.688, -4.208, -3.772],
        weights=[11.257, 9.485, 7.473, 8.707, 7.874],
    )
    resonator.log_out_spikes(-1)
    return resonator



def resonator_46_5():
    resonator = simple_resonator(
        freq0=46.58193456348157,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.059, -4.883, -4.745, -4.369],
        weights=[10.753, 6.776, 9.874, 9.674, 9.107],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_46():
    resonator = simple_resonator(
        freq0=46.020706436210695,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.782, -4.525, -4.206, -3.77],
        weights=[11.167, 9.198, 9.175, 8.52, 7.547],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_47_0():
    resonator = simple_resonator(
        freq0=47.011921651759856,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.633, -4.602, -4.947, -4.705],
        weights=[10.422, 7.805, 9.296, 9.894, 9.256],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_47_5():
    resonator = simple_resonator(
        freq0=47.59773999009954,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.515, -4.504, -4.947, -4.706],
        weights=[10.186, 7.569, 9.1, 9.894, 9.256],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_48_0():
    resonator = simple_resonator(
        freq0=48.046775273025006,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.75, -4.239, -4.716, -5.102],
        weights=[10.394, 8.847, 8.518, 9.549, 10.202],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_48_5():
    resonator = simple_resonator(
        freq0=48.50436360895858,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.75, -4.131, -4.599, -5.103],
        weights=[11.137, 9.302, 8.403, 9.423, 10.127],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_49():
    resonator = simple_resonator(
        freq0=49.12821394476512,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.75, -4.029, -4.569, -5.117],
        weights=[12.049, 10.63, 7.852, 9.189, 10.228],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_50():
    resonator = simple_resonator(
        freq0=50.09467061253099,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.546, -4.029, -4.439, -5.114],
        weights=[12.41, 10.444, 8.104, 9.106, 10.255],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_51():
    resonator = simple_resonator(
        freq0=51.09991483886941,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.852, -4.033, -4.197, -4.445],
        weights=[14.793, 12.173, 8.449, 9.028, 9.223],
    )
    resonator.log_out_spikes(-1)
    return resonator



def resonator_53():
    resonator = simple_resonator(
        freq0=53.05164769729845,
        clk_freq=1536000,
        lf=4,
        thetas=[-0.801, -4.826, -4.956, -5.553],
        weights=[11.669, 10.125, 9.491, 9.839, 10.887],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_54():
    resonator = simple_resonator(
        freq0=54.1804061589431,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.105, -5.002, -4.323, -5.113],
        weights=[14.269, 11.404, 10.186, 8.86, 10.12],
    )
    resonator.log_out_spikes(-1)
    return resonator



def resonator_56():
    resonator = simple_resonator(
        freq0=56.172332855963056,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.505, -5.251, -4.881, -5.671],
        weights=[14.709, 11.151, 10.67, 9.988, 11.327],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_57():
    resonator = simple_resonator(
        freq0=57.010725883664,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.973, -5.248, -4.874, -5.157],
        weights=[16.159, 11.349, 10.533, 9.938, 10.588],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_58():
    resonator = simple_resonator(
        freq0=58.09457998791617,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.373, -5.247, -4.87, -5.149],
        weights=[17.159, 11.823, 10.899, 10.329, 10.699],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_59():
    resonator = simple_resonator(
        freq0=59.22044394117036,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.717, -5.865, -5.252, -5.6],
        weights=[16.178, 10.47, 11.899, 10.579, 11.191],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_60():
    resonator = simple_resonator(
        freq0=60.15304935756674,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.718, -6.439, -5.349, -5.597],
        weights=[15.243, 9.82, 12.969, 10.767, 11.315],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_61():
    resonator = simple_resonator(
        freq0=61.115498147287816,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.718, -6.748, -5.652, -5.596],
        weights=[14.784, 9.42, 13.527, 11.397, 11.316],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_62():
    resonator = simple_resonator(
        freq0=62.10924608464209,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.718, -6.968, -5.858, -5.778],
        weights=[14.82, 9.48, 14.108, 11.915, 11.83],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_63():
    resonator = simple_resonator(
        freq0=63.13584519347914,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.718, -6.968, -5.858, -5.778],
        weights=[14.82, 9.48, 14.108, 11.915, 11.83],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_64():
    resonator = simple_resonator(
        freq0=64.19695183538636,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.345, -7.301, -6.315, -6.481],
        weights=[13.645, 9.549, 14.494, 12.8, 12.883],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_65():
    resonator = simple_resonator(
        freq0=65.01648739073171,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.297, -7.302, -6.438, -6.483],
        weights=[13.726, 9.475, 14.727, 13.351, 12.633],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_66():
    resonator = simple_resonator(
        freq0=66.14231401221625,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.011, -7.097, -6.242, -6.289],
        weights=[14.764, 10.93, 14.252, 12.492, 12.744],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_67():
    resonator = simple_resonator(
        freq0=67.01260761764014,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.806, -7.096, -6.559, -6.829],
        weights=[13.647, 10.327, 14.307, 13.485, 13.385],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_68():
    resonator = simple_resonator(
        freq0=68.209261325098,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.458, -7.109, -6.571, -6.841],
        weights=[14.667, 11.492, 13.857, 12.942, 13.647],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_69():
    resonator = simple_resonator(
        freq0=69.13517889964685,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.935, -7.109, -6.571, -6.469],
        weights=[15.997, 12.222, 14.232, 13.047, 12.988],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_70():
    resonator = simple_resonator(
        freq0=70.0865804441374,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.935, -7.282, -6.829, -6.646],
        weights=[15.207, 11.46, 14.72, 13.762, 13.346],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_71():
    resonator = simple_resonator(
        freq0=71.06453272940443,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.424, -7.284, -6.942, -6.95],
        weights=[14.644, 11.926, 14.535, 13.931, 13.941],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_72():
    resonator = simple_resonator(
        freq0=72.07016290953752,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.318, -7.285, -6.941, -6.95],
        weights=[14.959, 12.209, 14.569, 14.292, 13.84],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_73():
    resonator = simple_resonator(
        freq0=73.10466285560743,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.26, -7.29, -6.948, -6.96],
        weights=[16.145, 13.746, 14.678, 13.98, 13.568],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_74():
    resonator = simple_resonator(
        freq0=74.16929386806774,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.1, -6.827, -6.713, -6.512],
        weights=[20.303, 15.349, 14.156, 13.766, 13.136],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_75():
    resonator = simple_resonator(
        freq0=75.26539180700469,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.1, -6.827, -6.712, -6.51],
        weights=[20.303, 15.349, 14.156, 13.766, 13.136],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_76():
    resonator = simple_resonator(
        freq0=76.01430117821867,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.527, -7.502, -7.138, -6.972],
        weights=[19.343, 14.093, 15.141, 14.212, 13.944],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_77():
    resonator = simple_resonator(
        freq0=77.16603301425228,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.664, -7.836, -7.661, -7.215],
        weights=[18.075, 12.943, 15.851, 15.332, 14.544],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_78():
    resonator = simple_resonator(
        freq0=78.3532027529331,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.351, -7.697, -8.215, -7.54],
        weights=[17.646, 12.923, 15.453, 16.522, 15.025],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_79():
    resonator = simple_resonator(
        freq0=79.16515304052825,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.351, -7.697, -8.215, -7.54],
        weights=[17.646, 12.944, 15.485, 16.604, 15.105],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_80():
    resonator = simple_resonator(
        freq0=80.41512914116817,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.123, -7.442, -8.187, -8.014],
        weights=[18.013, 13.72, 14.973, 16.46, 16.173],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_81():
    resonator = simple_resonator(
        freq0=81.27060923841464,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.124, -7.443, -8.187, -8.013],
        weights=[17.91, 14.017, 15.01, 16.466, 16.405],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_82():
    resonator = simple_resonator(
        freq0=82.14448675710727,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.538, -8.039, -8.185, -7.864],
        weights=[18.86, 13.852, 16.259, 16.321, 15.813],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_83():
    resonator = simple_resonator(
        freq0=83.03736161316279,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.541, -7.328, -7.949, -8.854],
        weights=[20.822, 16.126, 14.262, 15.655, 17.573],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_84():
    resonator = simple_resonator(
        freq0=84.41367147415444,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.538, -7.325, -7.946, -8.853],
        weights=[21.506, 15.756, 14.819, 15.851, 17.436],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_85():
    resonator = simple_resonator(
        freq0=85.35684098783213,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.539, -7.324, -7.943, -8.847],
        weights=[21.624, 15.912, 15.221, 16.21, 17.81],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_86():
    resonator = simple_resonator(
        freq0=86.32132506679069,
        clk_freq=1536000,
        lf=4,
        thetas=[-3.349, -7.771, -8.471, -8.365],
        weights=[21.679, 15.042, 15.917, 17.148, 17.155],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_87():
    resonator = simple_resonator(
        freq0=87.30785449612546,
        clk_freq=1536000,
        lf=4,
        thetas=[-4.026, -9.325, -8.459, -8.083],
        weights=[21.719, 14.129, 18.742, 17.099, 16.313],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_88():
    resonator = simple_resonator(
        freq0=88.31719385446216,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.144, -8.859, -8.99, -8.747],
        weights=[18.526, 14.277, 17.641, 18.167, 17.425],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_89():
    resonator = simple_resonator(
        freq0=89.35014349018687,
        clk_freq=1536000,
        lf=4,
        thetas=[-1.931, -8.706, -8.89, -9.042],
        weights=[19.112, 14.838, 17.164, 17.909, 17.914],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_90():
    resonator = simple_resonator(
        freq0=90.40754163799971,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.113, -8.982, -8.803, -8.984],
        weights=[19.572, 15.247, 17.858, 17.907, 17.915],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_91():
    resonator = simple_resonator(
        freq0=91.4902666875566,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.379, -9.244, -8.691, -8.912],
        weights=[20.688, 15.664, 18.325, 17.551, 17.866],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_92():
    resonator = simple_resonator(
        freq0=92.04141287242139,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.431, -9.05, -9.005, -8.726],
        weights=[20.859, 16.269, 17.951, 18.168, 17.541],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_93():
    resonator = simple_resonator(
        freq0=93.16386912696314,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.12, -8.868, -9.126, -9.242],
        weights=[20.634, 16.057, 17.608, 18.369, 18.589],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_94():
    resonator = simple_resonator(
        freq0=94.3140403507528,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.542, -9.003, -8.841, -9.443],
        weights=[21.915, 16.963, 18.126, 17.553, 18.886],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_95():
    resonator = simple_resonator(
        freq0=95.4929658551372,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.804, -9.43, -9.342, -9.256],
        weights=[21.797, 16.375, 18.959, 18.54, 18.538],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_96():
    resonator = simple_resonator(
        freq0=96.09355054605001,
        clk_freq=1536000,
        lf=4,
        thetas=[-3.003, -9.549, -9.795, -9.763],
        weights=[21.551, 15.761, 18.866, 19.446, 19.352],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_97():
    resonator = simple_resonator(
        freq0=97.31767220905702,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.874, -9.884, -9.75, -9.763],
        weights=[21.391, 15.54, 19.713, 19.513, 19.466],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_98():
    resonator = simple_resonator(
        freq0=98.57338410852873,
        clk_freq=1536000,
        lf=4,
        thetas=[-3.142, -10.203, -9.525, -9.574],
        weights=[21.947, 16.219, 20.465, 19.194, 19.248],
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_99():
    resonator = simple_resonator(
        freq0=99.21347101832437,
        clk_freq=1536000,
        lf=4,
        thetas=[-2.581, -9.934, -10.257, -9.852],
        weights=[20.591, 15.903, 19.864, 20.538, 19.693],
    )
    resonator.log_out_spikes(-1)
    return resonator

# Dictionary mapping all frequencies (0-100 Hz) to their resonator functions

RESONATOR_FUNCTIONS = {
    0.100: resonator_0_100,
    0.250: resonator_0_250,
    0.372: resonator_0_372,
    0.694: resonator_0_694,
    0.898: resonator_0_898,
    1.00: resonator_1_00,
    1.30: resonator_1_30,
    1.59: resonator_1_59,
    1.75: resonator_1_75,
    1.90: resonator_1_90,
    1.95: resonator_1_95,
    2.50: resonator_2_50,
    2.68: resonator_2_68,
    2.79: resonator_2_79,
    3.05: resonator_3_05,
    3.39: resonator_3_39,
    3.47: resonator_3_47,
    3.72: resonator_3_72,
    5.09: resonator_5_09,
    5.26: resonator_5_26,
    5.45: resonator_5_45,
    5.87: resonator_5_87,
    6.36: resonator_6_36,
    6.94: resonator_6_94,
    7.63: resonator_7_63,
    8.98: resonator_8_98,
    9.54: resonator_9_54,
    10.051891142646022: resonator_10_0,
    10.118459958160233: resonator_10_1,
    10.254278212632183: resonator_10_2,
    10.323563876231049: resonator_10_3,
    10.464982559467092: resonator_10_4,
    10.537154852980658: resonator_10_5,
    10.61032953945969: resonator_10_6,
    10.75977080057884: resonator_10_7,
    11.15246316556347: resonator_11_1,
    11.23446657119261: resonator_11_2,
    11.317684842090335: resonator_11_3,
    11.4021451767328: resonator_11_4,
    11.574904952137842: resonator_11_5,
    11.663263005207597: resonator_11_6,
    11.752980412939964: resonator_11_7,
    11.844088788234073: resonator_11_8,
    11.93662073189215: resonator_11_9,
    12.030609871513349: resonator_12_0,
    12.126090902239644: resonator_12_1,
    12.223099629457563: resonator_12_2,
    12.321673013566091: resonator_12_3,
    12.421849216928416: resonator_12_4,
    12.523667653132748: resonator_12_5,
    12.627169038695829: resonator_12_6,
    12.732395447351628: resonator_12_7,
    12.839390367077272: resonator_12_8,
    12.948198760018604: resonator_12_9,
    13.058867125488849: resonator_13_0,
    13.17144356622582: resonator_13_1,
    13.285977858106046: resonator_13_2,
    13.40252152352803: resonator_13_4,
    13.521127908691994: resonator_13_5,
    13.641852265019601: resonator_13_6,
    13.764751834974732: resonator_13_7,
    13.88988594256541: resonator_13_8,
    14.017316088827481: resonator_14_0,
    14.14710605261292: resonator_14_1,
    14.279321997029863: resonator_14_2,
    14.414032581907502: resonator_14_4,
    14.551309082687574: resonator_14_5,
    14.691225516174953: resonator_14_6,
    14.833858773613548: resonator_14_7,
    14.833858773613548: resonator_14_8,
    14.97928876159015: resonator_14_9,
    15.127598551308864: resonator_15_1,
    15.278874536821952: resonator_15_2,
    15.433206602850456: resonator_15_4,
    15.590688302879546: resonator_15_5,
    15.751417048270055: resonator_15_6,
    15.751417048270055: resonator_15_7,
    15.915494309189533: resonator_15_9,
    16.083025828233634: resonator_16_0,
    16.254121847682928: resonator_16_2,
    16.428897351421455: resonator_16_3,
    16.428897351421455: resonator_16_4,
    16.607472322632557: resonator_16_6,
    16.789972018485663: resonator_16_7,
    16.9765272631355: resonator_16_9,
    17.167274760474104: resonator_17_0,
    17.167274760474104: resonator_17_1,
    17.362357428206767: resonator_17_3,
    17.561924754967762: resonator_17_4,
    17.561924754967762: resonator_17_5,
    17.76613318235111: resonator_17_6,
    17.97514651390818: resonator_17_9,
    18.189136353359466: resonator_18_1,
    18.40828257448428: resonator_18_2,
    18.40828257448428: resonator_18_3,
    18.40828257448428: resonator_18_4,
    18.632773825392626: resonator_18_5,
    18.632773825392626: resonator_18_6,
    18.86280807015056: resonator_18_8,
    19.098593171027442: resonator_18_9,
    19.098593171027442: resonator_19_0,
    19.340347514964495: resonator_19_1,
    19.340347514964495: resonator_19_2,
    19.340347514964495: resonator_19_3,
    19.588300688233275: resonator_19_4,
    19.588300688233275: resonator_19_5,
    19.842694203664873: resonator_19_6,
    19.842694203664873: resonator_19_8,
    20.103782285292045: resonator_19_9,
    20.103782285292045: resonator_20_1,
    20.371832715762604: resonator_20_2,
    20.371832715762604: resonator_20_3,
    20.647127752462097: resonator_20_4,
    20.647127752462097: resonator_20_6,
    20.929965118934184: resonator_20_7,
    20.929965118934184: resonator_20_8,
    20.929965118934184: resonator_20_9,
    21.51954160115768: resonator_21_3,
    21.51954160115768: resonator_21_4,
    21.51954160115768: resonator_21_5,
    21.82696362403136: resonator_21_6,
    21.82696362403136: resonator_21_7,
    21.82696362403136: resonator_21_8,
    22.143296430176743: resonator_21_9,
    22.143296430176743: resonator_22_1,
    22.46893314238522: resonator_22_2,
    22.46893314238522: resonator_22_3,
    22.46893314238522: resonator_22_4,
    22.8042903534656: resonator_22_5,
    22.8042903534656: resonator_22_6,
    22.8042903534656: resonator_22_7,
    22.8042903534656: resonator_22_8,
    23.149809904275685: resonator_23_1,
    23.505960825879928: resonator_23_2,
    23.505960825879928: resonator_23_3,
    23.505960825879928: resonator_23_4,
    23.505960825879928: resonator_23_5,
    23.8732414637843: resonator_23_6,
    23.8732414637843: resonator_23_7,
    23.8732414637843: resonator_23_8,
    24.25218180447929: resonator_23_9,
    24.25218180447929: resonator_24_0,
    24.25218180447929: resonator_24_2,
    24.643346027132182: resonator_24_5,
    24.643346027132182: resonator_24_6,
    25.047335306265495: resonator_24_7,
    25.047335306265495: resonator_24_8,
    25.047335306265495: resonator_25_0,
    25.464790894703256: resonator_25_1,
    25.464790894703256: resonator_25_2,
    25.464790894703256: resonator_25_3,
    25.464790894703256: resonator_25_4,
    25.89639752003721: resonator_25_5,
    25.89639752003721: resonator_25_6,
    25.89639752003721: resonator_25_7,
    25.89639752003721: resonator_25_8,
    26.34288713245164: resonator_25_9,
    26.34288713245164: resonator_26_0,
    26.34288713245164: resonator_26_1,
    26.34288713245164: resonator_26_3,
    26.80504304705606: resonator_26_4,
    26.8: resonator_26_8,
    27.9: resonator_27_9,
    28.8: resonator_28_8,




    31.30916913283187: resonator_31_3,
    31.83098861837907: resonator_31_8,
    32.16605165646727: resonator_32_1,
    32.508243695365856: resonator_32_5,
    32.85779470284291: resonator_32_8,
    33.432985857378455: resonator_33_4,
    33.72819986053411: resonator_33_7,
   

    34.028673801385196: resonator_34,
    35.20478003876026: resonator_35_2,
    35.61509216042413: resonator_35_6,

    36.90549405029457: resonator_36_9,
    36.03508145476876: resonator_36,
    37.35666145922237: resonator_37_3,
    37.81899637827216: resonator_37_8,

    38.485830067561594: resonator_38_4,
    38.680695029929: resonator_38_6,
    38.877543350691994: resonator_38_8,
 
    39.68538840732975: resonator_39_6,
    39.78873577297384: resonator_39_7,
    40.102032905044496: resonator_40_0,
    40.52751866531022: resonator_40_5,
    41.2942555049242: resonator_41_2,

    42.3237521795622: resonator_42_3,
    42.559539099782604: resonator_42_5,
    43.03908320231536: resonator_43_0,

    44.54482372251298: resonator_44_5,
 
    45.20377081899986: resonator_45_2,
    45.60858070693121: resonator_45_5,
    45.88250611658244: resonator_45_8,
    46.020706436210695: resonator_45_9,

    46.58193456348157: resonator_46_5,
    46.020706436210695: resonator_46,
    47.011921651759856: resonator_47_0,
    47.59773999009954: resonator_47_5,
    48.046775273025006: resonator_48_0,
    48.50436360895858: resonator_48_5,
    49.12821394476512: resonator_49,
    50.09467061253099: resonator_50,
    51.09991483886941: resonator_51,

    53.05164769729845: resonator_53,
    54.1804061589431: resonator_54,

    56.172332855963056: resonator_56,
    57.010725883664: resonator_57,
    58.09457998791617: resonator_58,
    59.22044394117036: resonator_59,
    60.15304935756674: resonator_60,
    61.115498147287816: resonator_61,
    62.10924608464209: resonator_62,
    63.13584519347914: resonator_63,
    64.19695183538636: resonator_64,
    65.01648739073171: resonator_65,
    66.14231401221625: resonator_66,
    67.01260761764014: resonator_67,
    68.209261325098: resonator_68,
    69.13517889964685: resonator_69,
    70.0865804441374: resonator_70,
    71.06453272940443: resonator_71,
    72.07016290953752: resonator_72,
    73.10466285560743: resonator_73,
    74.16929386806774: resonator_74,
    75.26539180700469: resonator_75,
    76.01430117821867: resonator_76,
    77.16603301425228: resonator_77,
    78.3532027529331: resonator_78,
    79.16515304052825: resonator_79,
    80.41512914116817: resonator_80,
    81.27060923841464: resonator_81,
    82.14448675710727: resonator_82,
    83.03736161316279: resonator_83,
    84.41367147415444: resonator_84,
    85.35684098783213: resonator_85,
    86.32132506679069: resonator_86,
    87.30785449612546: resonator_87,
    88.31719385446216: resonator_88,
    89.35014349018687: resonator_89,
    90.40754163799971: resonator_90,
    91.4902666875566: resonator_91,
    92.04141287242139: resonator_92,
    93.16386912696314: resonator_93,
    94.3140403507528: resonator_94,
    95.4929658551372: resonator_95,
    96.09355054605001: resonator_96,
    97.31767220905702: resonator_97,
    98.57338410852873: resonator_98,
    99.21347101832437: resonator_99,
}



# Function to get the closest available resonator.

def get_closest_resonator(target_freq):

    """

    Returns the resonator function with frequency closest to target_freq

    """

    available_freqs = list(RESONATOR_FUNCTIONS.keys())

    closest_freq = min(available_freqs, key=lambda x: abs(x - target_freq))

    return RESONATOR_FUNCTIONS[closest_freq], closest_freq



# Function to get resonator in a specific frequency band

def get_resonators_in_band(f_min, f_max):

    """

    Returns a list of (freq, resonator_func) for all resonators in the specified band

    """

    return [(f, RESONATOR_FUNCTIONS[f]) for f in sorted(RESONATOR_FUNCTIONS.keys())

            if f_min <= f <= f_max]

