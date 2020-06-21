import os
import sys
import csv
import math
from random import randint

from clear_cut.utils.graph_tools import GraphTools


class ImageUtils(object):

    debug = False
    _graph_tools = None

    def __init__(self):
        # increase csv file size limit
        csv.field_size_limit(sys.maxsize)

    @property
    def graph_tools(self):
        if not self._graph_tools:
            self._graph_tools = GraphTools(debug=self.debug)
        
        return self._graph_tools

    def reduce_iter(self, i):
        return i - (i > 5) * 8

    def edge_kill(self, edg_img, coord, radius, wipe=False):
        """
        Determine whether the "shells border" are all non-edge pixels or not
        """
        # This may have already been wiped
        if not edg_img[coord[0], coord[1]]:
            if wipe:
                return edg_img

            return True

        # Initial counter and pre-define useful values
        border_size = 2 * radius + 1

        # Run over the square of pixels surrounding "radius" pixels around coord
        for i in range(0, border_size**2):
            dx = (i % border_size) - radius
            dy = (i // border_size) - radius

            x = coord[0] + dx
            y = coord[1] + dy

            if wipe:
                # Wipe whole shell of edge pixels
                try:
                    edg_img[x, y] = 0
                except IndexError:
                    # The central edge pixel is too close to the image perimeter, so ignore it
                    pass

                continue

            # Skip the inner shell pixels, just run around the border
            if abs(dx) != radius and abs(dy) != radius:
                continue
            
            try:
                if edg_img[x, y] > 0.:
                    # Found an edge pixel on the border, thus we cannot wipe this shell
                    return False
                
            except IndexError:
                # The central edge pixel is too close to the image perimeter, so ignore it
                continue

        if wipe:
            return edg_img

        # If we got here, all border pixels must be non-edge pixels
        return True

    def edge_killer(self, edge_image, pixel_tolerance=1):
        """
        Check and wipe out any edge pixels found within a pixel_tolerance radius.
        We refer to the radius of surround pixels as "the shell".
        """
        self.graph_tools._print_if_debugging(f'Reducing image noise with pixel tolerance of {pixel_tolerance} ...')

        edge_coordinates = self.graph_tools.edge_pixel_positions(edge_image)

        for edge_coordinate in edge_coordinates:
            # Iterate over each layer of the shell (r --> r - 1 decrements)
            for sub_radius in reversed(range(1, pixel_tolerance + 1)):
                noisy_shell_found = self.edge_kill(edge_image, edge_coordinate, radius=sub_radius)

                if noisy_shell_found:
                    edge_image = self.edge_kill(edge_image, edge_coordinate, radius=sub_radius-1, wipe=True)
                    break
        
        self.graph_tools._print_if_debugging(f'... image reduced of noise.')
        return edge_image

    def __within_radius(self, x, y, chosen_one, R=10):
        return math.sqrt((x - chosen_one[0]) ** 2 + (y - chosen_one[1]) ** 2) <= R
