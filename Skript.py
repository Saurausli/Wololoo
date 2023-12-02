import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
from sklearn.linear_model import LinearRegression
"""
Fehlerhafte Sensoren:
a1f343aa-dbbd-4b7e-ac42-58ee4bfffd06
96e70afa-3ce0-4674-a331-8b85aed97068
"""

def create_regression(sampled,min_offset=0,ax = None):
    X = np.arange(len(sampled)).reshape(-1,1)
    y = sampled
    model = LinearRegression().fit(X,y)
    y_get = model.predict(X)
    
    if ax is not None:
        X = X + min_offset
        ax.plot(X,y_get, label = "model")
    return model

def butter_lowpass_filter(data, cutoff, fs = 30.0, order = 10):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def resample(df):
    """
    Resamples the DataFrame to have a uniform time interval of 10 minutes.
    Interpolates missing values using linear interpolation.
    """
    # Convert the 'Gemessen am' column to datetime and set it as the index
    df['Gemessen am'] = pd.to_datetime(df['Gemessen am'], utc=True)
    df.set_index('Gemessen am', inplace=True)
    df.sort_index(inplace=True)
    # Extract the 'Füllstandsdistanz' column and resample with a 10-minute interval
    series = df['Füllstandsdistanz'].resample('10T')
    # Calculate the mean for each 10-minute interval
    series = series.mean()
    # Interpolate missing values using linear interpolation
    series = series.interpolate(method='linear')
    # Create a new DataFrame with the resampled and interpolated values
    result = pd.DataFrame({"Füllstandsdistanz": series})
    # Reset the index to make the time column a regular column
    result.reset_index(inplace=True)
    
    return result


def diff_dynamic_range(diff):
    """
    Compares positive peaks with negative peaks.
    """
    window_min = 200
    window_max = 200
    found = np.zeros_like(diff)
    
    for i in range(0, len(diff), 1):
        # Create a window for comparison
        if i < window_min:
            window = diff[0: window_max + i]
        elif i > len(diff) - window_max:
            window = diff[i - window_min: len(diff)]
        else:
            window = diff[i - window_min: window_max + i]

        # Check if the current point is the minimum in the window and below a threshold
        if diff[i] == np.min(window) and diff[i] < -0.25:
            max_value = np.max(window)
            if max_value < 0:
                found[i] = np.abs(diff[i]) / np.abs(max_value)
            else:
                found[i] = 1
        else:
            found[i] = 0

    # Thresholding the found values
    found[found < 1] = 0
    found[found >= 1] = 1
    return found


activate_plot = True

def plot_data(axs,sampled,filtered,device_name,negativeDiff,found,diff):
    
    axs[0].plot(sampled)
    axs[0].set_ylabel("Füllstand (%)")
    axs[0].set_title(f"Resampelte Daten {device_name}")
    # axs[0].legend()
    axs[1].plot(filtered, label = "filtered")
    axs[1].set_title(f"Gefilterte Daten {device_name}")
    axs[1].set_ylabel("Füllstand (%)")
    # axs[1].legend()
    axs[2].set_title(f"Ableitung mit offset")
    axs[2].plot(diff, label = "diff")
    axs[2].plot(negativeDiff, label = "negative diff")
    axs[3].plot(found, label = "dynamic")
    axs[3].set_title(f"Entleerungsdetektion")

if __name__ == "__main__":
    df = pd.read_csv("Data/Data/fullstandssensoren-sammelstellen-stadt-stgallen.csv",sep=";")
    sorted_data = {}
    # sorted_data = df.groupby("Device ID")
    sorted_data = df.groupby("Device ID")
    device_values = {"mean": []}
    device_list = []
    for device_name, device_df in sorted_data:
        device_df[device_df['Füllstandsdistanz']>1800] = pd.NA
        device_df = device_df.dropna()
        device_df = device_df[['Füllstandsdistanz','Gemessen am']]
        unsampled = device_df['Füllstandsdistanz'].to_numpy()
        device_df = resample(device_df)
        sampled = device_df['Füllstandsdistanz'].to_numpy()
        sampled = -sampled
        
        #sampled = (sampled - min_full)/(max_full-min_full)*100
        filtered = butter_lowpass_filter(sampled,1.5, 144, 5)
        sampled = (sampled - np.min(filtered)) /(np.max(filtered)- np.min(filtered))*100
        filtered = (filtered - np.min(filtered)) /(np.max(filtered)- np.min(filtered))*100
        
        diff = filtered[1:] - filtered[0:-1]
        found = diff_dynamic_range(diff)
        negativeDiff = diff.copy()
        diff[diff<0.1] = 0
        negativeDiff[negativeDiff>-0.1] = 0
        indices = list(np.where(found == 1))[0]
        
        
        if activate_plot == True:
            fig, axs = plt.subplots(5, figsize = (16,9), sharex= True)
            plot_data(axs,sampled,filtered,device_name,negativeDiff,found,diff)
            # axs[3].legend()
        b = []
        
        if len(indices) > 1:
            for i in range(0,len(indices)-1,1):
                marge = 30
                if activate_plot == True:
                    model = create_regression(filtered[indices[i] + marge :indices[i+1]- marge],indices[i] +marge,axs[0])
                else:
                    model = create_regression(filtered[indices[i] + marge :indices[i+1]- marge],indices[i] +marge)
                b.append(model.coef_[0])
            b = np.array(b)
            b = b * 6 * 24
            b = 100/b
            if np.var(b) < 100:
                print(f"{device_name} {np.mean(b):.3f} {np.var(b):.3f}")
                device_values["mean"].append(np.mean(b))
            # device_values["var"].append(np.var(b))
                device_list.append(device_name)


            if activate_plot == True:
                ticks = (indices[1:]-indices[:-1])//2+indices[:-1]

                axs[4].plot(ticks,b,marker="o",linestyle="")
                ticks = np.linspace(0,len(found),10)
                date_ticks=device_df.iloc[ticks]['Gemessen am'].dt.strftime('%d-%m-%Y')
                axs[4].hlines(np.mean(b),0,len(found))
            # axs[4].set_xticks(ticks,date_ticks)
                axs[4].set_ylim(0,np.max(b)*1.4)
                axs[4].set_title(f"Anstieg des Füllstandes")
                axs[4].set_ylabel(f"Anstieg des Füllstandes (%/Tag)")
        if activate_plot == True:
            fig.savefig(f"auto_plots/{device_name}.png")
            #plt.show()
        # plt.hist(df["Füllstandsdistanz"],bins=20)
    x = np.arange(len(device_list))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in device_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    # ax.set_ylabel('Length (mm)')
    ax.set_title("Anstieg des Füllstandes pro Container")
    # ax.set_ylabel(f"Anstieg des Füllstandes\n (%/Tag)")
    ax.set_ylabel(f"Füllzeit\n (Tag)")
    ax.set_xticks(x + width, device_list,rotation=90)
    ax.legend(loc='upper left', ncols=3)
    ax.grid()
    #plt.show()
    fig.savefig(f"auto_plots/{device_name}.png")

