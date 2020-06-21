# Clear Cut

This package is intended to serve as a tool to sketch out the edges of any provided image. This is a personal project which I am - to use an appropriate term - *channeling* my number crunching desire into.

Refer to [this repo](https://github.com/chrispdharman/clear-cut) if you would like to play around with the source code.

# To run locally

1. Pick a new folder location, e.g. `~/Desktop/mock`
1. In a terminal, go into this folder `cd [absolute_path]/mock/`
1. Create a new Python3 virtual environment `python3 -m venv venv`
1. Activate this virtual environment: `source venv/bin/activate`
1. Install the latest (or specific) verion of the ClearCut python module: `pip install clear-cut(==1.3.1)`
1. Run Python in shell: `python`
1. Excecute the following Python script (tailoring the paths to your local set up):
```
from clear_cut.clear_cutter import ClearCut

image_filepath = '/Users/christopherharman/Desktop/mock/clear-cut-mock-logo.jpg'
results_path = '/Users/myusername/Desktop/mock/'

# All kwargs are optional
#   param debug: toggles on/off output to terminal
#   param results_path: sets results directory. Omitting this parameter will write results to current working directory
#   param image_filepath: location to image. Omitting this parameter will use a default Bob Ross image instead
clear_cut = ClearCut(debug=True, image_filepath=image_filepath, results_path=results_path)
clear_cut.run()
```

On completion, you should have a set of ClearCut processed images in a `/results/` directory
