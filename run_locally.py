from clear_cut.clear_cutter import ClearCut

my_image = '/Users/christopherharman/Desktop/mock/clear-cut-mock-logo.jpg'
clear_cut = ClearCut(debug=True, serverless=False, image_filepath=my_image)
clear_cut.run()
