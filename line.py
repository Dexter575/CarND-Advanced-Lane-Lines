import numpy as np

class Line(object):
    def __init__(self):
        self.detected = False

        # Store the x values of the last n fits of the line
        self.recent_xfitted = []
        
        # Store average of x values of the fitted line over the last n iteration
        self.bestx = None
        
        # Store Average of polyminal coefficients over the last n iterations
        self.best_fit = None
        
        self.current_fit = [np.array([False])]
        self.current_x = None
        self.radius_of_curvature = None
        self.line_base_pos = None