from bokeh.plotting import figure, show, output_file, gridplot
import colorsys
import numpy as np
import pcl
import os

# Set parameters
plot_width = 2000
plot_height = 2000

# import data and preprocess using point cloud
datafile = '\\final_project_point_cloud.fuse'
pcdOutlierfile = '\\outlier.pcd'
pcdInlierfile = '\\inliers.pcd'

# Load file
#points_array = np.loadtxt(datafile, dtype=np.float32, usecols=(0,1,2))

def generatePcl(points_array):
    point_cloud = pcl.PointCloud(points_array)
    # Transform and create pcd file
    fil = point_cloud.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(0.3)
    
    fil.set_negative(False)
    # Inlier
    pcl.save(fil.filter(), pcdInlierfile)

    # Outlier
    fil.set_negative(True)
    pcl.save(fil.filter(), pcdOutlierfile)
    

    return 0

# Functions used to read and convert data into float and arrays.

# Read the fuse file line after line
def processFile(filename):
    fullArray = []
    with open(filename) as f:
        if (os.path.splitext(filename)[1]) == ".pcd":
            for _ in xrange(11):
                next(f)
        
        if (os.path.splitext(filename)[1]) == ".pcd":    
            for line in f:
                lat, lon, elev = line.strip().split(" ")
                lineArray = [float(lat), float(lon), float(elev)]
                fullArray.append(lineArray)
            
        for line in f:
            lat, lon, elev, intensity = line.strip().split(" ")
            lineArray = [float(lat), float(lon), float(elev)]
            fullArray.append(lineArray)    
            
        return fullArray


# Select a column from a multidimensional array
def column(matrix, i):
    return [row[i] for row in matrix]


def generatePlot(array, minimum, maximum):
    x = np.array(column(array, 0))
    y = np.array(column(array, 1))
    z = np.array(column(array, 2))

    print("bounding values: %s,%s,%s,%s" %(min(x), min(y), max(x), max(y)))

    colors = define_color(z, minimum, maximum)

    fig = figure(plot_width = plot_width, plot_height = plot_height)
    fig.scatter(x, y, fill_alpha=0.6, color=colors, line_color=None)
    
    return fig


def renderPlot(s1, s2, s3):
    p = gridplot(([[s1, s2, s3]]))
    output_file("\\DTM_Scatter.html", title="Digital Terrain Model")
    print("Opening Browser")
    show(p)  # open a browser
    return 0


def define_color(z, minimum, maximum):
    colors = []

    h = (0.8 - (i - minimum) * 0.8 / (maximum - minimum) for i in z)
    i = 0
    for j in h:
        c = colorsys.hsv_to_rgb(j, 1, 1)
        c_rgb = "#%02x%02x%02x" % (c[0] * 255, c[1] * 255, c[2] * 255)
        colors.append(c_rgb)
        i += 1
    return colors


def main():

    array = processFile(datafile)   # Read data file
    
    # Set variables minimum and maximum to keep the same color scale.
    minimum = min(np.array(column(array, 2)))
    maximum = max(np.array(column(array, 2)))                       
                       
    s1 = generatePlot(array, minimum, maximum)

    generatePcl(array)              # Generate pcl file
    
    pclOutarray = processFile(pcdOutlierfile) # Read outlier pcl file
    s2 = generatePlot(pclOutarray, minimum, maximum)
    
    pclInarray = processFile(pcdInlierfile) # Read inlier pcl file
    s3 = generatePlot(pclInarray, minimum, maximum)
    
    renderPlot(s1, s2, s3)

if __name__ == '__main__':
    main()


