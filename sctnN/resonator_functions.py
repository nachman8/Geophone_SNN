import numpy as np

from sctnN.resonator import simple_resonator




###########################################

# Resonator functions for 10-100 Hz range

###########################################



def resonator_10_5():

    resonator = simple_resonator(

        freq0=10.5,

        clk_freq=153600,

        lf=4,

        thetas=[-11.912, -11.103, -9.652, -9.996],

        weights=[42.046, 18.636, 21.913, 19.553, 20.16],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_11_0():

    resonator = simple_resonator(

        freq0=11.0,

        clk_freq=153600,

        lf=4,

        thetas=[-12.327, -11.735, -9.997, -10.622],

        weights=[43.742, 19.585, 23.06, 20.334, 21.4],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_11_5():

    resonator = simple_resonator(

        freq0=11.5,

        clk_freq=153600,

        lf=4,

        thetas=[-13.016, -12.335, -10.737, -10.707],

        weights=[46.083, 20.491, 24.346, 21.682, 21.655],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_12_8():

    resonator = simple_resonator(

        freq0=12.8,

        clk_freq=153600,

        lf=4,

        thetas=[-14.609, -13.618, -11.787, -12.061],

        weights=[51.511, 22.744, 26.914, 23.81, 24.306],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_13_0():

    resonator = simple_resonator(

        freq0=13.0,

        clk_freq=153600,

        lf=4,

        thetas=[-38.705, -14.082, -11.844, -12.294],

        weights=[97.872, 21.307, 27.448, 24.055, 24.879],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_15_9():

    resonator = simple_resonator(

        freq0=15.9,

        clk_freq=153600,

        lf=4,

        thetas=[-18.833, -17.091, -14.972, -15.48],

        weights=[63.287, 26.537, 33.677, 30.133, 31.076],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_16_0():

    resonator = simple_resonator(

        freq0=16.0,

        clk_freq=153600,

        lf=4,

        thetas=[-19.184, -17.323, -15.152, -15.803],

        weights=[63.808, 26.451, 34.141, 30.497, 31.678],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_16_6():

    resonator = simple_resonator(

        freq0=16.6,

        clk_freq=153600,

        lf=4,

        thetas=[-20.0, -17.797, -15.956, -16.104],

        weights=[66.282, 27.3, 35.106, 32.001, 32.329],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_17_5():

    resonator = simple_resonator(

        freq0=17.5,

        clk_freq=153600,

        lf=4,

        thetas=[-24.442, -17.659, -17.053, -16.969],

        weights=[78.467, 29.833, 35.414, 34.069, 33.882],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_19_0():

    resonator = simple_resonator(

        freq0=19.0,

        clk_freq=153600,

        lf=4,

        thetas=[-22.972, -20.365, -18.31, -18.796],

        weights=[75.971, 31.307, 40.053, 36.719, 37.694],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_19_5():

    resonator = simple_resonator(

        freq0=19.5,

        clk_freq=153600,

        lf=4,

        thetas=[-23.292, -20.967, -18.739, -19.253],

        weights=[77.357, 32.09, 41.255, 37.596, 38.596],

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

def resonator_22_1():

    resonator = simple_resonator(

        freq0=22.1,

        clk_freq=153600,

        lf=4,

        thetas=[-26.511, -23.624, -21.416, -21.755],

        weights=[87.574, 36.152, 46.334, 42.9, 43.672],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_25_0():

    resonator = simple_resonator(

        freq0=25.0,

        clk_freq=153600,

        lf=4,

        thetas=[-30.365, -26.786, -24.465, -24.437],

        weights=[99.455, 40.538, 52.741, 48.947, 49.019],

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



def resonator_30_5():

    resonator = simple_resonator(

        freq0=30.5,

        clk_freq=153600,

        lf=4,

        thetas=[-37.124, -32.549, -29.814, -29.008],

        weights=[123.204, 50.851, 64.454, 59.25, 58.383],

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



def resonator_33_9():

    resonator = simple_resonator(

        freq0=33.9,

        clk_freq=153600,

        lf=4,

        thetas=[-45.112, -36.415, -32.509, -32.316],

        weights=[144.962, 56.675, 72.227, 64.768, 64.959],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_34_7():

    resonator = simple_resonator(

        freq0=34.7,

        clk_freq=153600,

        lf=4,

        thetas=[-44.809, -37.682, -34.249, -33.187],

        weights=[142.565, 55.948, 73.91, 68.354, 67.02],

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



def resonator_37_2():

    resonator = simple_resonator(

        freq0=37.2,

        clk_freq=153600,

        lf=4,

        thetas=[-48.663, -40.06, -35.864, -35.336],

        weights=[157.321, 62.183, 79.307, 71.374, 71.063],

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



def resonator_40_2():

    resonator = simple_resonator(

        freq0=40.2,

        clk_freq=153600,

        lf=4,

        thetas=[-52.983, -43.606, -39.073, -37.241],

        weights=[170.532, 67.256, 86.596, 77.271, 75.25],

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

        freq0=41.2,

        clk_freq=153600,

        lf=4,

        thetas=[-54.501, -44.59, -39.86, -38.707],

        weights=[175.469, 69.046, 88.335, 79.108, 78.08],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_43_6():

    resonator = simple_resonator(

        freq0=43.6,

        clk_freq=153600,

        lf=4,

        thetas=[-58.654, -47.541, -42.627, -40.103],

        weights=[187.055, 72.849, 94.515, 84.064, 81.135],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_46_2():

    resonator = simple_resonator(

        freq0=46.2,

        clk_freq=153600,

        lf=4,

        thetas=[-61.279, -49.554, -45.289, -44.119],

        weights=[194.982, 75.689, 97.964, 90.261, 89.009],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_47_7():

    resonator = simple_resonator(

        freq0=47.7,

        clk_freq=153600,

        lf=4,

        thetas=[-62.781, -51.17, -46.907, -45.359],

        weights=[200.524, 78.397, 101.374, 92.831, 91.37],

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



def resonator_50_9():

    resonator = simple_resonator(

        freq0=50.9,

        clk_freq=153600,

        lf=4,

        thetas=[-69.695, -55.164, -50.732, -47.095],

        weights=[219.153, 83.76, 109.522, 99.543, 95.232],

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



def resonator_52_6():

    resonator = simple_resonator(

        freq0=52.6,

        clk_freq=153600,

        lf=4,

        thetas=[-78.345, -57.279, -52.931, -48.217],

        weights=[239.144, 86.311, 114.419, 103.404, 97.135],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_54_5():

    resonator = simple_resonator(

        freq0=54.5,

        clk_freq=153600,

        lf=4,

        thetas=[-81.139, -59.604, -54.623, -49.983],

        weights=[247.52, 89.124, 119.23, 106.752, 100.593],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_58_7():

    resonator = simple_resonator(

        freq0=58.7,

        clk_freq=153600,

        lf=4,

        thetas=[-88.619, -64.356, -59.071, -53.569],

        weights=[268.708, 95.611, 128.823, 115.302, 107.954],

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

def resonator_63_6():

    resonator = simple_resonator(

        freq0=63.6,

        clk_freq=153600,

        lf=4,

        thetas=[-98.054, -70.205, -64.087, -57.686],

        weights=[294.356, 103.068, 140.527, 124.915, 116.543],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_69_4():

    resonator = simple_resonator(

        freq0=69.4,

        clk_freq=153600,

        lf=4,

        thetas=[-113.694, -77.332, -69.207, -62.432],

        weights=[334.262, 112.405, 154.573, 135.118, 126.891],

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



def resonator_76_3():

    resonator = simple_resonator(

        freq0=76.3,

        clk_freq=153600,

        lf=4,

        thetas=[-127.835, -85.872, -76.422, -67.925],

        weights=[371.729, 122.996, 171.202, 148.778, 139.294],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_89_8():

    resonator = simple_resonator(

        freq0=89.8,

        clk_freq=153600,

        lf=4,

        thetas=[-106.516, -93.167, -90.835, -81.611],

        weights=[359.026, 152.001, 182.454, 179.91, 167.481],

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



def resonator_95_4():

    resonator = simple_resonator(

        freq0=95.4,

        clk_freq=153600,

        lf=4,

        thetas=[-117.549, -100.74, -96.393, -85.843],

        weights=[387.993, 160.185, 197.341, 190.822, 177.199],

    )

    resonator.log_out_spikes(-1)

    return resonator



# Dictionary mapping all frequencies (0-100 Hz) to their resonator functions

RESONATOR_FUNCTIONS = {


    # 10-100 Hz range

    10.5: resonator_10_5,

    11.0: resonator_11_0,

    11.5: resonator_11_5,

    12.8: resonator_12_8,

    13.0: resonator_13_0,

    15.9: resonator_15_9,

    16.0: resonator_16_0,

    16.6: resonator_16_6,

    17.5: resonator_17_5,

    19.0: resonator_19_0,

    19.5: resonator_19_5,

    20.103782285292045: resonator_20_1,

    22.1: resonator_22_1,

    25.0: resonator_25_0,

    26.8: resonator_26_8,

    27.9: resonator_27_9,

    28.8: resonator_28_8,

    30.5: resonator_30_5,

    33.72819986053411: resonator_33_7,

    33.9: resonator_33_9,

    34.7: resonator_34_7,

    36.03508145476876: resonator_36,

    37.2: resonator_37_2,

    39.68538840732975: resonator_39_6,

    40.2: resonator_40_2,

    40.52751866531022: resonator_40_5,

    41.2: resonator_41_2,

    43.6: resonator_43_6,

    46.2: resonator_46_2,

    47.7: resonator_47_7,

    48.046775273025006: resonator_48_0,

    50.9: resonator_50_9,

    51.09991483886941: resonator_51,

    52.6: resonator_52_6,

    54.5: resonator_54_5,

    58.7: resonator_58_7,

    63.13584519347914: resonator_63,

    63.6: resonator_63_6,

    69.4: resonator_69_4,

    70.0865804441374: resonator_70,

    76.3: resonator_76_3,

    89.8: resonator_89_8,

    91.4902666875566: resonator_91,

    95.4: resonator_95_4

}



# Function to get the closest available resonator

def get_closest_resonator(target_freq):
    """
    Returns the resonator function with frequency closest to target_freq
    """
    # Handle both dictionary and set cases
    if isinstance(RESONATOR_FUNCTIONS, dict):
        available_freqs = list(RESONATOR_FUNCTIONS.keys())
        closest_freq = min(available_freqs, key=lambda x: abs(x - target_freq))
        return RESONATOR_FUNCTIONS[closest_freq], closest_freq
    else:
        # If it's a set, try direct function name lookup
        resonator_name = f"resonator_{str(target_freq).replace('.', '_')}"
        import sys
        current_module = sys.modules[__name__]
        if hasattr(current_module, resonator_name):
            resonator_func = getattr(current_module, resonator_name)
            return resonator_func, target_freq
        else:
            # Fallback to using any function
            func = next(iter(RESONATOR_FUNCTIONS))
            return func, target_freq

# Function to get resonator in a specific frequency band

def get_resonators_in_band(f_min, f_max):

    """

    Returns a list of (freq, resonator_func) for all resonators in the specified band

    """

    return [(f, RESONATOR_FUNCTIONS[f]) for f in sorted(RESONATOR_FUNCTIONS.keys())

            if f_min <= f <= f_max]
