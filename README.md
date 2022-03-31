# image-processing

##functions:

###imReadAndConvert: reads on image in pyhton  as well as converts it to a gray scale image or rgb depending on user input in the function

###imDisplay: displays the image in python in a window

###transformfromRGB2YIQ: transfroms the images pixel values fom (r,g,b) (red, green , blue) to YIQ values and returns the numpy dot product of this new image pixels

###transformfromYIQ2RGB: transfroms the images pixel values fom (Y,I,Q) to (r,g,b) (red, green , blue) values and returns the numpy dot product of this new image pixels

###hsitogramEqualize: creats a 1D graph made up of bins showing the distrebution of pixel values normalized and shows a graph with the corrosponding values , also shows after eqalizing the new better resulted images histogram

###quantizeImage: this function takes in a paramter and finds the best values to reduce the images color by. it also shows the error by computig this loss and reduces the images to nQuant colors 
