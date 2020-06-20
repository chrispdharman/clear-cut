from clear_cut.clear_cutter import ClearCut

my_image = '/Users/christopherharman/Desktop/mock/clear-cut-mock-logo.jpg'
results_path = '/Users/christopherharman/Desktop/mock/'
clear_cut = ClearCut(debug=True, serverless=False, image_filepath=my_image, results_path=results_path)
clear_cut.run()
