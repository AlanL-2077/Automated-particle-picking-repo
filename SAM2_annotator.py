##########
# The annotator to get the V3 mask
# This file generates one type of file
# The .npy, the final mask for image segmentation
##########

# import the packages
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from pathlib import Path
import sys
import cv2
import torch

# This program is run on an individual virtual environment, with isolated configuration of libraries
# The image predictor from SAM2(segment anything model 2) is the core package
from sam2.sam2_image_predictor import SAM2ImagePredictor

# The labeling work will be an interactive process
# Thus, we import the button to get button controls
import matplotlib
from matplotlib.widgets import Button
# Then, specify the rendering backend as 'TkAgg'
# it can create an independent window, supporting interactive operations
matplotlib.use('TkAgg')


class SAM2Annotator:
    def __init__(self, predictor):
        """
        The initialization of the annotator class.

        argument:
        predictor: the model to use for prediction
        """
        self.predictor = predictor
        # get the device type: returns 'cuda' or 'cpu'
        self.device = str(predictor.device).split(':')[0]

        # store the coordinates (x,y) for every user click
        self.points = []
        # the model needs to know which point is a "foreground point" and which one is "background"
        # so it stores '1' or '0' for each coordinate in self.points
        self.labels = []
        # it stores the accepted mask for each single segment operation in one picture
        self.annotations = []

        # element interface part
        # top container, everything (plots, buttons, operations) will be stored here
        self.fig = None
        # the info (like dictionary) of each axes(subplot) will be stored here
        self.axes = None

        # batch process part
        # to show which file is now processing
        self.current_file_index = 0
        # all the file pairs(.mrc, .npy) that will be processed
        self.file_list = []

    def load_data(self, mrc_path, v2_mask_path, downsample_factor = 2):
        """
        load the mrc and v2_mask data

        argument:
        mrc_path: path to the mrc file
        v2_mask_path: path to the v2_mask NPY file
        downsample_factor: downsampling factors, default is 2
        """
        # record the input path of mrc and mask
        self.mrc_path = mrc_path
        self.v2_mask_path = v2_mask_path

        # record the downsample factor
        self.downsample_factor = downsample_factor

        # load the mrc file
        print(f"\nLoading MRC file: {Path(mrc_path).name}")
        with mrcfile.open(mrc_path, permissive = True) as mrc:
            self.mrc_data_full = mrc.data.astype(np.float32)

        # min-max normalization
        # then to [0,255]
        # then convert to uint8
        self.mrc_data_full = ((self.mrc_data_full - self.mrc_data_full.min()) /
                              (self.mrc_data_full.max() - self.mrc_data_full.min()) * 255).astype(np.uint8)

        # downsampling
        # to increase the processing speed
        if downsample_factor > 1:
            # get the new (W, H)
            new_size = (self.mrc_data_full.shape[1] // downsample_factor,
                        self.mrc_data_full.shape[0] // downsample_factor)
            print(f"Downsampling from {self.mrc_data_full.shape} to {new_size}")

            # downsize plot with INTER_AREA
            self.mrc_data = cv2.resize(self.mrc_data_full, new_size, interpolation = cv2.INTER_AREA)
        else: # no need downsampling
            self.mrc_data = self.mrc_data_full

        # to RGB because of the SAM2 requirement
        self.rgb_image = np.stack([self.mrc_data] * 3, axis = -1)

        # load the V2 mask
        print(f"Loading V2 mask...")
        self.v2_mask_full = np.load(v2_mask_path)

        # perform the downsampling to V2 mask as well
        if downsample_factor > 1:
            new_size = (self.v2_mask_full.shape[1] // downsample_factor,
                        self.v2_mask_full.shape[0] // downsample_factor)
            # use the interpolation of INTER_NEAREST with the nearest label directly
            # V2 mask includes 0,1,2,3,4,-1
            self.v2_mask = cv2.resize(self.v2_mask_full.astype(np.float32),
                                      new_size, interpolation = cv2.INTER_NEAREST).astype(self.v2_mask_full.dtype)
        else: # no downsampling
            self.v2_mask = self.v2_mask_full

        # store the name of mrc
        self.mrc_name = Path(mrc_path).stem
        # copy the mask after downsampling, good habit
        self.final_mask = self.v2_mask.copy()

        # reset all lists
        self.points = []
        self.labels = []
        self.annotations = []
        self.current_mask = None

        # configure the SAM2
        print("Setting image in SAM2...")
        # config device
        if self.device == 'cuda':
            # introduce the autocast
            # convert the float32 to 16 automatically if needed
            with torch.autocast('cuda', dtype = torch.float16):
                # set_image() to input the picture
                self.predictor.set_image(self.rgb_image)
        else:
            self.predictor.set_image(self.rgb_image)
        # for now, the annotation is good to go
        print(f"Ready for annotation")

    def setup_ui(self):
        """
        create the interactive user interface

        components: a set of subplots, showing the labeling process
                    buttons: accept, clear, undo, save, skip, quit functions

        """
        # the whole UI size, 16 * 8
        self.fig = plt.figure(figsize = (16, 8))

        # there will be 6 subplots in total, 2 rows and 3 columns
        self.ax_mrc = plt.subplot(2, 3, 1) # original mrc

        self.ax_sam = plt.subplot(2, 3, 2) # the result of sam2 prediction

        self.ax_final = plt.subplot(2, 3, 3) # the visible result of the final mask

        self.ax_v2 = plt.subplot(2, 3, 4) # the original v2

        self.ax_ignore = plt.subplot(2, 3, 5) # intro
        self.ax_ignore.axis('off')
        self.ax_ignore.text(0.01, 0.5, 'Welcome to the cell annotator powered by SAM2', transform = self.ax_ignore.transAxes,
                           fontsize = 10, verticalalignment = 'top', fontfamily = 'monospace')

        self.ax_info = plt.subplot(2, 3, 6) # the information of each operation
        self.ax_info.axis('off')

        # generate the buttons
        self.setup_controls()

        # display the picture in the squares, respectively
        self.update_display()

        # introduce the 'mpl_connect'
        # it can connect events to the callback functions
        # left click and right click
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # the keyboard input
        self.cid2 = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def setup_controls(self):
        """
        set the button controls

        7 buttons are needed, including
        'accept': accept the current mask prediction
        'clear': clear all the points in the mrc
        'Undo Point': delete the last point
        ‘Undo Object’: delete the last accepted label
        'save': save the current mask and go to the next
        'skip': skip the current picture to the next
        'quit': quit the labeling process
        """

        # define buttons
        # each one will be ('name', position, meaning)
        buttons = [('Accept (A)', 0.15, self.accept_mask),
                   ('Clear (C)', 0.23, self.clear_points),
                   ('Undo Pt (X)', 0.31, self.undo_point),
                   ('Undo Obj (Z)', 0.39, self.undo_object),
                   ('Save&Next (S)', 0.47, self.save_and_next),
                   ('Skip (K)', 0.55, self.skip_image),
                   ('Quit (Q)', 0.63, self.quit_app)]

        # the width of the button is 7%
        button_width = 0.07
        # the height of the button is 3%
        button_height = 0.03
        # the distance to the bottom edge is 2% of the height
        y_position = 0.02

        self.buttons = {}
        for name, x_position, callback in buttons:
            # create the button zone, with the position and the size
            ax = plt.axes([x_position, y_position, button_width, button_height])
            # create a button with its name
            btn = Button(ax, name)
            # link the button to its callback functions
            btn.on_clicked(callback)
            # save that to the self.buttons
            self.buttons[name] = btn

    def update_display(self):
        """
        display the image in the beginning

        besides, update the image with each operation
        """
        # the original mrc file
        self.ax_mrc.clear() # clear all points
        self.ax_mrc.imshow(self.mrc_data, cmap = 'gray') # show the mrc plot

        # the process of the batch files
        if self.file_list:
            # display processing progress
            progress = f"[{self.current_file_index + 1}/{len(self.file_list)}]"
        else:
            progress = ""

        # set the title of the mrc plot
        self.ax_mrc.set_title(f'Click to annotate {progress}')
        self.ax_mrc.axis('off')

        # display all points according to their labels
        for (x, y), label in zip(self.points, self.labels):
            # label == 1 means the point is foreground
            # 0 is a background point
            # lime for foreground and red for the background
            color = 'lime' if label == 1 else 'red'
            # draw the point on mrc
            self.ax_mrc.plot(x, y, 'o', color = color, markersize = 10,
                             markeredgecolor = 'white', markeredgewidth = 2)

        # display the v2 mask, on the lower left
        # clear first
        self.ax_v2.clear()
        # display the original mrc data with 70% opacity
        self.ax_v2.imshow(self.mrc_data, cmap = 'gray', alpha = 0.7)
        # print the existed labels with show_v2_overlay()
        self.show_v2_overlay(self.ax_v2)
        # set the title
        self.ax_v2.set_title('V2 mask (Red: correct cells, Blue: border)')
        # no axis
        self.ax_v2.axis('off')

        # display the final mask
        # clear first
        self.ax_final.clear()
        # display the original mrc data with 70% opacity
        self.ax_final.imshow(self.mrc_data, cmap = 'gray', alpha = 0.7)
        # print the final mask
        self.show_final_overlay(self.ax_final)
        # set the title, how many attempts in total
        self.ax_final.set_title(f'Final result (New objects: {len(self.annotations)})')
        # no axis
        self.ax_final.axis('off')

        # update the info, in the lower right square
        self.update_info()

        # draw all subplots
        plt.draw()

    def show_v2_overlay(self, ax):
        """
        extract the v2 mask and show it on the ax_v2
        """
        # create a full-zero array with the same MRC scale
        overlay = np.zeros((*self.mrc_data.shape, 4))

        # the correct particle printing
        # V2 contains the label of the right particle as 1/2/3/4
        particle_mask = (self.v2_mask > 0) & (self.v2_mask <= 4)
        # paint a pure red with 50% opacity
        overlay[particle_mask] = [1, 0, 0, 0.5]

        # CAUTION!
        # The black border is remapped to 6 in V4 mask, because of the model training feedback
        # Thus, this code cannot be run on V4 but on V2 only

        # besides, the label for the black border
        # these are labeled as -1
        border_mask = self.v2_mask == -1
        # the border will be painted with sky blue, with 50% opacity
        overlay[border_mask] = [0.3, 0.5, 1, 0.5]

        # display the mask to the ax_v2
        ax.imshow(overlay)

    def show_final_overlay(self, ax):
        """
        display the final mask

        because all labels generated by this program are -1,
        thus its outcomes will be combined with the V2 for the final mask
        """
        # create a full-zero array
        overlay = np.zeros((*self.mrc_data.shape, 4))

        # the correct cell printing, same as above
        cell_mask = (self.final_mask > 0) & (self.final_mask <= 4)
        # paint a pure red with alpha of 0.5
        overlay[cell_mask] = [1, 0, 0, 0.5]

        # how to calculate the new -1 area ?
        # first, we extract all pixels that are labeled -1
        ignore_mask = self.final_mask == -1
        # secondly, the only area labeled '-1' in the V2 mask is the black border
        original_border = (self.v2_mask == -1)

        # thus, we use ~original_border to invert the V2 mask, get the bright area
        # then, we use '&' to get the new -1 area
        # this is the new -1 area without the black border
        new_regions = ignore_mask & ~original_border
        # paint the black border with sky blue
        overlay[original_border] = [0.3, 0.5, 1, 0.5]
        # paint the new -1 area with yellow
        overlay[new_regions] = [1, 1, 0, 0.5]
        # display the mask to ax_final
        ax.imshow(overlay)

    def update_info(self):
        """
        update the dynamic info

        the info square is located in the lower right corner
        """
        # clear first
        self.ax_info.clear()
        self.ax_info.axis('off')

        # the name of mrc file
        info_text = f"File: {self.mrc_name} \n\n"

        # the progress of the batch process
        if self.file_list:
            info_text += f"Progress one the current file \n"
            # the index of this file
            info_text += f"Current: {self.current_file_index + 1} / {len(self.file_list)} \n"
            # how many files are completed for now
            info_text += f"Completed: {self.current_file_index} \n"
            # how many files are waited to be processed
            info_text += f"Remaining: {len(self.file_list) - self.current_file_index - 1} \n\n"

        # the information about the V2 mask
        info_text += "=== Level 2 Statistics === \n"

        # how many pixels in total
        # 2048 X 2048 = 4,194,304
        total_pixels = self.v2_mask.size
        # how many pixels are particles
        correct_cells = np.sum((self.v2_mask > 0) & (self.v2_mask <= 4))
        # how many pixels are recognized as the black border
        black_border = np.sum(self.v2_mask == -1)

        # the number and percentage of particle pixels
        info_text += f"Correct cells: {correct_cells:,} ({correct_cells / total_pixels * 100:.1f}%) \n"
        # the number and percentage of black border pixels
        info_text += f"Black border: {black_border:,} ({black_border / total_pixels * 100:.1f}%)\n\n"

        # the info about operations done with SAM2
        info_text += "=== New Annotations === \n"
        # how many times of annotations are done
        info_text += f"New operations: {len(self.annotations)} \n"

        # this program will only give -1 label
        # but in V2 they were 0
        if self.annotations:
            new_pixels = np.sum((self.final_mask == -1) & (self.v2_mask == 0))
            # the number and percentage of new labeled objects
            info_text += f"New pixels: {new_pixels:,} ({new_pixels / total_pixels * 100:.1f}%)\n"

        # the information about points
        # 1: a foreground point
        # 0: a background point
        if self.points:
            info_text += f"\n=== Current points === \n"
            info_text += f"Total: {len(self.points)} \n" # how many points in total
            info_text += f"Positive: {sum(1 for l in self.labels if l == 1)} \n"
            info_text += f"Negative: {sum(1 for l in self.labels if l == 0)} \n"

        # detect the downsampling
        if self.downsample_factor > 1:
            # show the downsampling factor
            info_text += f"\n[The current downsampling factor is {self.downsample_factor}.]"

        # here we input the info to ax_info square,
        # use transAxes, with range of [0, 1] x [0, 1]
        # monospaced font
        self.ax_info.text(0.1, 0.9, info_text, transform = self.ax_info.transAxes,
                          fontsize = 10, verticalalignment = 'top', fontfamily = 'monospace')

    def on_click(self, event):
        """
        process the click operation

        left click: add a foreground point
        right click: add a background point
        """
        # first, the clicking operation must take place in the ax_mrc only
        # introduce inaxes to ensure no response on invalid clicking
        if event.inaxes != self.ax_mrc:
            return

        # invalid x or y will not be recorded
        if event.xdata is None or event.ydata is None:
            return

        # shift the coordinate to the int
        x, y = int(event.xdata), int(event.ydata)

        # add the coordinate to self.points
        self.points.append([x, y])
        # add the binary label of that coordinate, foreground or back
        self.labels.append(1 if event.button == 1 else 0)

        # print the info
        print(f"Added {'positive' if self.labels[-1] == 1 else 'negative'} point at ({x}, {y})")

        # predict the area with SAM2
        self.predict_current()

        # display the visible outcome to the axes
        self.update_display()

    def predict_current(self):
        """
        ***********
        CORE PART
        ***********

        predict the mask via SAM2 on the basis of clicked points and their labels

        the outcome will be stored and displayed in the ax_sam
        """
        # if points are recorded
        if not self.points:
            return

        try:
            # convert the list to numpy array
            scaled_points = np.array(self.points).astype(np.float32)

            # model prediction
            if self.device == 'cuda':
                # use autocast, float32 to 16 if poosible
                with torch.autocast('cuda', dtype = torch.float16):
                    with torch.no_grad():
                        # input the coordinates and their labels
                        # use multimask_output to return 3 predictions with the confidence score
                        masks, scores, _ = self.predictor.predict(point_coords = scaled_points,
                                                                  point_labels = np.array(self.labels),
                                                                  multimask_output = True)
            else: # cpu
                masks, scores, _ = self.predictor.predict(point_coords = scaled_points,
                                                          point_labels = np.array(self.labels),
                                                          multimask_output = True)

            # choose the best prediction based on score
            best_idx = np.argmax(scores)
            # the best prediction as mask
            # save the confidence score
            self.current_mask = masks[best_idx].astype(bool)
            self.current_score = scores[best_idx]

            # place this prediction in ax_sam
            # clear first
            self.ax_sam.clear()
            # show the original mrc file first, with 80% opacity
            self.ax_sam.imshow(self.mrc_data, cmap = 'gray', alpha = 0.8)

            # create a full-zero array
            mask_overlay = np.zeros((*self.mrc_data.shape, 4), dtype = np.float32)
            # paint the prediction with cyan, 50% opacity
            mask_overlay[self.current_mask] = [0, 1, 1, 0.5]

            # display the visible prediction to ax_sam
            self.ax_sam.imshow(mask_overlay)
            # show the title with confidence score
            self.ax_sam.set_title(f'SAM2 Prediction (Score: {self.current_score:.3f})')
            self.ax_sam.axis('off')

            # show the changes
            plt.draw()

        # errors found during predicting
        except Exception as e:
            print(f"Prediction error: {e}")
            # traceback the problem
            import traceback
            traceback.print_exc()

    def undo_point(self, event):
        """
        undo the last point from ax_mrc

        no matter if it's a foreground or a background point
        """
        # there must have points in ax_mrc
        if self.points:
            # remove the last point
            removed_point = self.points.pop()
            removed_label = self.labels.pop()

            # which type this point is
            point_type = "positive" if removed_label == 1 else "negative"
            # print the undo information
            print(f"Removed {point_type} point at ({removed_point[0]}, {removed_point[1]})")

            # is it the last point
            # no
            if self.points:
                # get the new prediction
                self.predict_current()
            # yes
            else:
                # clean the prediction
                self.current_mask = None
                # clear the ax_sam
                self.ax_sam.clear()
                # show the plain mrc
                self.ax_sam.imshow(self.mrc_data, cmap = 'gray')
                self.ax_sam.set_title('SAM2 Prediction')
                self.ax_sam.axis('off')

        # update the display
        self.update_display()

    def accept_mask(self, event):
        """
        trigger the event of the button 'Accept(a)'
        """
        # at least one prediction mask is recorded
        if self.current_mask is not None:
            # input the prediction info to self.annotations
            self.annotations.append({'mask': self.current_mask.copy(),
                                     'score': self.current_score,
                                     'points': self.points.copy(),
                                     'labels': self.labels.copy()})
            # print the info
            print(f"Accepted object (confidence score: {self.current_score:.3f})")

            # the combination of prediction mask and V2 mask
            update_region = self.current_mask & (self.v2_mask == 0)
            # label the new with -1
            self.final_mask[update_region] = -1

            # clear the points
            self.clear_points(None)
            # update the display, basically the ax_mrc
            self.update_display()

    def clear_points(self, event):
        """
        clean all points and their labels
        """
        # clean the points
        self.points = []
        # clean the labels
        self.labels = []
        # clean the prediction mask
        self.current_mask = None

        # clear the ax_mrc
        self.ax_sam.clear()
        # show the mrc again
        self.ax_sam.imshow(self.mrc_data, cmap = 'gray')
        self.ax_sam.set_title('SAM2 Prediction')
        self.ax_sam.axis('off')

        # update
        self.update_display()

    def undo_object(self, event):
        """
        remove the last prediction mask from ax_final
        """
        # at least one accepted mask are recorded in self.annotations
        if self.annotations:
            # remove the last prediction
            removed = self.annotations.pop()
            print(f"Removed last object annotation")
            # because an accepted mask is removed, the final mask should be recalculated
            self.recalculate_final_mask()
            # show the changes
            self.update_display()

    def recalculate_final_mask(self):
        """
        recalculate the final mask

        Two steps to take this:
        1. Start from V2 mask
        2. Find if there is other annotations
        3. If there is/are, implement it/them
        """
        # start from V2
        self.final_mask = self.v2_mask.copy()
        # if there are accepted annotations
        for ann in self.annotations:
            # get the new labelled regions
            update_region = ann['mask'] & (self.v2_mask == 0)
            # label this region with -1
            self.final_mask[update_region] = -1

    def save_and_next(self, event):
        """
        save all predictions of this mrc and move to the next image
        """
        self.save_current()
        self.next_image()

    def save_current(self):
        """
        save the final mask
        """
        # find which stage this V2 mask belongs to
        stage = None
        for s in ['stage1', 'stage2', 'stage3', 'stage4']:
            if s in str(self.v2_mask_path).lower():
                stage = s
                break

        # create the stage folder, for the first file
        output_dir = Path("outputs") / stage
        output_dir.mkdir(parents = True, exist_ok = True)

        # if downsampling is implemented
        if hasattr(self, 'downsample_factor') and self.downsample_factor > 1:
            print(f"Upsampling and merging masks...")

            # use the original V2 mask (4096 * 4096)
            final_mask_full = self.v2_mask_full.copy()

            # upsampling each annotations
            for ann in self.annotations:
                # upsampling to the original size
                mask_full = cv2.resize(ann['mask'].astype(np.uint8),
                                       (self.mrc_data_full.shape[1], self.mrc_data_full.shape[0]),
                                       interpolation = cv2.INTER_NEAREST).astype(bool)

                # the combination of mask_full and V2 mask
                update_region = mask_full & (self.v2_mask_full == 0)
                # label them with '-1'
                final_mask_full[update_region] = -1

        else: # no downsampling
            final_mask_full = self.final_mask

        # save the final mask as V3 mask
        output_path = output_dir / f"mask_{self.mrc_name}_v3.npy"
        # save the file
        np.save(output_path, final_mask_full)
        # print the saving info
        print(f"Saved merged mask to: {output_path}")

        # generate a summary of this V3 mask
        self.print_final_statistics(final_mask_full)

    def skip_image(self, event):
        """
        skip the current mrc file
        """
        print(f"Skipped {self.mrc_name}")
        # load the next mrc and V2 mask
        self.next_image()

    def next_image(self):
        """
        move to the next mrc file and V2 mask
        """
        # if we haven't reached the final mrc
        if self.file_list and self.current_file_index < len(self.file_list) - 1:
            # add one on index
            self.current_file_index += 1
            # get the path from the file list based on the index
            mrc_path, v2_path = self.file_list[self.current_file_index]

            # load the mrc and V2 mask based on the path, with the downsampling factor
            self.load_data(mrc_path, v2_path, self.downsample_factor)
            # show the image
            self.update_display()
        else: # this is the final file
            print("\nAll files processed!")
            # quit
            self.quit_app(None)

    def on_key(self, event):
        """
        apart from the clicking, we can also 'press' the button with keyboard
        """
        # a for accept mask
        if event.key == 'a':
            self.accept_mask(None)
        # c for clear all points in ax_mrc
        elif event.key == 'c':
            self.clear_points(None)
        # x for undo the last points
        elif event.key == 'x':
            self.undo_point(None)
        # z for undo the last accepted mask
        elif event.key == 'z':
            self.undo_object(None)
        # s for save the current mask and move to the next
        elif event.key == 's':
            self.save_and_next(None)
        # k for skip the current mask
        elif event.key == 'k':
            self.skip_image(None)
        # q for quit the program
        elif event.key == 'q':
            self.quit_app(None)

    def print_final_statistics(self, final_mask):
        """
        print the statistic of the V3 mask
        """
        print("\n=== Final Statistics ===")
        # get the total pixels
        total_pixels = final_mask.size

        # for each category
        # 1/2/3/4: correct stage particlees
        correct_cells = np.sum((final_mask > 0) & (final_mask <= 4))
        # -1: ignored objects
        ignored_objects = np.sum(final_mask == -1)
        # 0: the background
        background = np.sum(final_mask == 0)

        # the black border in V2
        original_border = np.sum((self.v2_mask_full == -1))
        # how many new pixels are labeled as -1
        new_ignored = ignored_objects - original_border

        # display the information
        print(f"Total pixels: {total_pixels:,}")
        print(f"Correct cells (1-4): {correct_cells:,} ({correct_cells / total_pixels * 100:.1f}%)")
        print(f"Black border (original): {original_border:,} ({original_border / total_pixels * 100:.1f}%)")
        print(f"New ignored objects: {new_ignored:,} ({new_ignored / total_pixels * 100:.1f}%)")

    def quit_app(self, event):
        """
        quit the program
        """
        # close the UI
        plt.close('all')
        # end the program
        sys.exit(0)

def get_file_pairs(dataset_dir, v2_dir):
    """
    get the path of mrc and V2 mask
    save them by pairs
    """
    # use a list to store
    file_pairs = []

    # define the name mapping
    stage_map = {'stage1': 'stageI',
                 'stage2': 'stageII',
                 'stage3': 'stageIII',
                 'stage4': 'stageIV'}

    # find the stage folder, respectively
    for v2_stage, mrc_stage in stage_map.items():
        mrc_file_path = Path(dataset_dir) / mrc_stage
        v2_file_path = Path(v2_dir) / v2_stage

        # if the folder are not found
        if not mrc_file_path.exists() or not v2_file_path.exists():
            print(f"Could not find {mrc_stage} or {v2_stage} directories")
            continue

        # get all the mrc files
        mrc_files = sorted(mrc_file_path.glob('*.mrc'))

        # for each mrc file
        for mrc_file in mrc_files:
            # compose the name of its corresponding V2 mask
            v2_file_name = f"mask_{mrc_file.stem}_v2.npy"
            # find the mask
            v2_file = v2_file_path / v2_file_name

            # pair them and store to the list
            if v2_file.exists():
                file_pairs.append((mrc_file, v2_file))
            else: # V2 mask doesn't found
                print(f"Warning: Could not find V2 mask for {mrc_file.name}")

    return file_pairs


def main():
    """
    main function

    Customized function according to the user's dataset location
    """
    print("Initializing SAM2 for batch processing...")

    # we use the tiny version of SAM2.1-hiera
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
    # print the predictor
    print(f"SAM2 loaded (device: {predictor.device})")

    # construct the annotator
    annotator = SAM2Annotator(predictor)

    # Path configuration
    # for both mrc and V2 mask
    dataset_dir = r"C:\Users\13146\OneDrive\Desktop\Bristol\Final Project\Dataset-processed"
    v2_dir = r"C:\Users\13146\OneDrive\Desktop\Bristol\Final Project\Image segmentation Level 2"

    # get all the files
    print("\n Scanning for files...")
    file_pairs = get_file_pairs(dataset_dir, v2_dir)

    # print the number of found pairs
    print(f"Found {len(file_pairs)} file pairs to process")

    # introduce a function to choose how many files to process
    response = input("\n Process all files? (y/n) or enter number to process: ")
    if response.lower() == 'n': # no process
        return
    elif response.isdigit(): # the input number
        # set a limit
        file_pairs = file_pairs[:int(response)]
        # print how many files to process
        print(f"Processing first {len(file_pairs)} files")

    # file pairs to the file list
    annotator.file_list = file_pairs
    # zero the index
    annotator.current_file_index = 0

    # load the first file pair
    mrc_path, v2_path = file_pairs[0]
    # load the data with the downsampling factor
    annotator.load_data(mrc_path, v2_path, downsample_factor = 4)

    # batch annotation start
    print("\n === Batch Processing Start ===")

    # set up the user interface
    annotator.setup_ui()
    # show the plot
    plt.show()

if __name__ == "__main__":
    main()