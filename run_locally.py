from clear_cut.clear_cutter import ClearCut

my_image = '/Users/christopherharman/Documents/Images/clear-cut-mock-logo.jpg'
clear_cut = ClearCut(debug=True, image_filepath=my_image)
clear_cut.run()
