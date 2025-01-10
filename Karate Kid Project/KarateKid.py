import cv2
import pyautogui
import numpy as np
from collections import deque
import random
import time
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plt

class KarateKidAutomation :
    def __init__(self):
        self.trunk_template = cv2.imread('trunk.png', cv2.IMREAD_COLOR)
        self.branch_templates = [
            cv2.imread("branch1.png"),
            cv2.imread("branch2.png"),
            cv2.imread("branch3.png"),
            cv2.imread("branch4.png"),
            cv2.imread("branch5.png"),
            cv2.imread("branch6.png")
        ]
        self.level_template = cv2.imread('level.png')
        self.starting_screen = cv2.imread('starting_screen.png') 
        self.play_template = cv2.imread("play_template.png")
        self.blue_power_boost_template = cv2.imread("blue_powerboost.png")
        bottom_right = False
        bottom_left = False

    def starting_screen_detection(self):
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        game_window = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        reference_gray = cv2.cvtColor(self.starting_screen, cv2.COLOR_BGR2GRAY)
        game_gray = cv2.cvtColor(game_window, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()

        keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_gray, None)
        keypoints_frame, descriptors_frame = sift.detectAndCompute(game_gray, None)

        good_matches = []
        for i, ref_desc in enumerate(descriptors_ref):
            distances = np.linalg.norm(descriptors_frame - ref_desc, axis=1)
            min_distance_idx = np.argmin(distances)  
            if distances[min_distance_idx] < 0.7 * np.partition(distances, 2)[1]:
                good_matches.append(cv2.DMatch(i, min_distance_idx, distances[min_distance_idx]))

        if len(good_matches) > 20:
            print("Starting screen detected!")

            ref_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            frame_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC)

            h, w = reference_gray.shape
            corners_ref = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(corners_ref, H)

            cv2.polylines(game_window, [np.int32(transformed_corners)], True, (255, 0, 0), 3)

            x_min, y_min = map(int, np.int32(transformed_corners.min(axis=0)).flatten())
            x_max, y_max = map(int, np.int32(transformed_corners.max(axis=0)).flatten())
        
            return (x_min, y_min), (x_max, y_max)
        else:
            print("Starting screen not detected!")
            return None,None

    def click_play_button(self,frame,frame_offset):
        w, h, _ = self.play_template.shape
        threshold = 0.8

        result = cv2.matchTemplate(frame, self.play_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            screen_click_x = top_left[0] + frame_offset[0] 
            screen_click_y = top_left[1] + frame_offset[1]
            print(f"Clicking play button at: ({screen_click_x}, {screen_click_y})")
            pyautogui.click(screen_click_x, screen_click_y)
            time.sleep(0.1)
        else:
            print("Play button not detected.")

    def press_random_direction(self):
        keyboard = Controller()

        direction = random.choice(["left", "right"])
        print(f"Pressing {direction} key...")

        if direction == "left":
            keyboard.press(Key.left)
            time.sleep(0.4)
            keyboard.release(Key.left)
        else:
            keyboard.press(Key.right)
            time.sleep(0.4)
            keyboard.release(Key.right)
        return direction
        
    def game_window(self):
        top_left, bottom_right = self.starting_screen_detection()

        if top_left is None or bottom_right is None:
            return None ,None

        x1, y1 = top_left
        x2, y2 = bottom_right

        print(f"Play area coordinates: Top-left ({x1}, {y1}), Bottom-right ({x2}, {y2})")
        screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
        play_frame = np.array(screenshot)
        play_frame = cv2.cvtColor(play_frame, cv2.COLOR_RGB2BGR)

        self.click_play_button(play_frame,top_left)
        init_pos = self.press_random_direction()
        return init_pos , (x1, y1, x2 - x1, y2 - y1)

    def capture_frames(self):
        init_pos , region = self.game_window()
        if init_pos is None or region is None :
            print("Cannot capture frames. Play area not detected.")
            return
        screenshot = pyautogui.screenshot(region=region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        trunk_top_left, trunk_bottom_right = self.trunk_detection(frame)
        
        if trunk_top_left and trunk_bottom_right:
            print(f"Trunk detected at: Top-left {trunk_top_left}, Bottom-right {trunk_bottom_right}")
        else:
            print("Trunk not detected after first click.")
            return
        current_pos = init_pos
        while True:
            screenshot = pyautogui.screenshot(region=region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, trunk_top_left, trunk_bottom_right, (0, 255, 0), 2)

            current_pos = self.play(frame,trunk_top_left, trunk_bottom_right,current_pos)
            if (self.level_progress(frame)):
                print("Waiting ...")
                time.sleep(5)
            
            cv2.imshow("Play Area Stream", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def trunk_detection(self,frame):
        if self.trunk_template is None or frame is None:
            print("Error: Template or frame not loaded properly.")
            return None, None

        frame_height, frame_width = frame.shape[:2]

        scaling_factor = 0.9 
        min_scale = 0.5
        threshold = 0.7

        current_scale = 1.0
        while current_scale >= min_scale:
            scaled_width = int(self.trunk_template.shape[1] * current_scale)
            scaled_height = int(self.trunk_template.shape[0] * current_scale)
            resized_template = cv2.resize(self.trunk_template, (scaled_width, scaled_height))

            if scaled_width > frame_width or scaled_height > frame_height:
                current_scale *= scaling_factor
                continue

            result = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                top_left = max_loc
                bottom_right = (top_left[0] + scaled_width, top_left[1] + scaled_height)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                print(f"Trunk detected at scale {current_scale:.2f}, location: {top_left}")
                
                return top_left, bottom_right

            current_scale *= scaling_factor

        print("Trunk not detected at any scale.")
        return None, None

    def detect_branch(self,frame,threshold=0.8):
        # Probable values : None -> no branch above the player , Right , Left .
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_height, frame_width = gray_frame.shape[:2]
        midpoint_y = frame_height // 2
        midpoint_x = frame_width // 2
        for template in self.branch_templates:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.normalize(gray_template, None, 0, 255, cv2.NORM_MINMAX)
            template_h, template_w = gray_template.shape[:2]

            result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)

            if not isinstance(threshold, (float, int)):
                raise ValueError("Threshold must be a scalar value (e.g., 0.8)")

            loc = np.where(result >= threshold)

            for pt in zip(*loc[::-1]):  
                x, y = pt
                # detected_boxes.append((x, y, x + template_w, y + template_h))
                if y < midpoint_y:
                    if x < midpoint_x :
                        cv2.rectangle(frame, (x, y), (x + template_w, y + template_h), (200, 255, 100), 1)
                        return 'left'
                    else :
                        cv2.rectangle(frame, (x, y), (x + template_w, y + template_h), (200, 255, 100), 1)
                        return 'right'
                elif y > midpoint_y :
                    if x < midpoint_x :
                        cv2.rectangle(frame, (x, y), (x + template_w, y + template_h), (200, 255, 100), 1)
                        return 'bottom_left'
                    else :
                        cv2.rectangle(frame, (x, y), (x + template_w, y + template_h), (200, 255, 100), 1)
                        return 'bottom_right'
                # cv2.rectangle(frame, (x, y), (x + template_w, y + template_h), (200, 255, 100), 1)
        return 'None'

    def trunk_number_detection(self,frame, trunk_top_left, trunk_bottom_right):
        roi = frame[(trunk_top_left[1]+trunk_bottom_right[1])//2:trunk_bottom_right[1], trunk_top_left[0]:trunk_bottom_right[0]]

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mean = self.calculate_mean_intensity(roi,lower_yellow,upper_yellow)
        if int(mean) >= 17 :
            return 3
        elif int(mean) >= 13 :
            return 2
        elif int(mean) >= 11 :
            return 4
        else : 
            return -1

    def calculate_mean_intensity(self,image, lower_yellow, upper_yellow):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mean_intensity = np.mean(yellow_mask)
        return mean_intensity

    def glass_detection(self,frame,trunk_top_left,trunk_bottom_right):
        roi_glass = frame[trunk_top_left[1]+72:trunk_bottom_right[1]-15,trunk_top_left[0]-12:trunk_bottom_right[0]+12,0]

        _, thresh_image = cv2.threshold(roi_glass, 200, 255, cv2.THRESH_BINARY)
        n = cv2.countNonZero(thresh_image)
        # print(n) 
        if 1000 > n > 150 :
            return True
        else :
            return False

    def green_power_boost_detection(self,frame):
        frame_height, frame_width = frame.shape[:2]
        midpoint_y = frame_height // 3
        midpoint_x = frame_width // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 50, 50])  
        upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = h / float(w)

            if 0.8 < aspect_ratio < 2.5 and w > 15 and h > 15:
                if y < midpoint_y:
                    if x < midpoint_x :
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 244, 55), 1)
                        return 'left'
                    else :
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 244, 55), 1)
                        return 'right'
        return 'None'

    def blue_power_boost_detection(self,frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

        frame_height, frame_width = gray_frame.shape[:2]
        midpoint_y, midpoint_x = frame_height // 2, frame_width // 2

        gray_template = cv2.cvtColor(self.blue_power_boost_template, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.normalize(gray_template, None, 0, 255, cv2.NORM_MINMAX)
        template_h, template_w = gray_template.shape[:2]

        result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.8)

        for pt in zip(*loc[::-1]):
            x, y = pt
            if y < midpoint_y:
                if x < midpoint_x:
                    cv2.rectangle(frame, (x, y), (x + template_w, y + template_h), (255, 0, 0), 2)
                    return "left"
                else:
                    cv2.rectangle(frame, (x, y), (x + template_w, y + template_h), (255, 0, 0), 2)
                    return "right"

        return "None"

    def level_progress (self,frame):
        w, h ,_= self.level_template.shape

        threshold = 0.8

        result = cv2.matchTemplate(frame, self.level_template, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            return True
        return False

    def play (self,frame,trunk_top_left, trunk_bottom_right,current_pos):
        keyboard = Controller()
        roi_frame = frame[trunk_top_left[1]:trunk_bottom_right[1],trunk_top_left[0]//3+50:trunk_top_left[0]*2-60]
        # self.blue_power_boost_detection(roi_frame)
        # Glass Detection Level 3
        glass = self.glass_detection(frame,trunk_top_left,trunk_bottom_right)
        if glass :
            print("Glass detected !")
            if current_pos == 'left' :
                keyboard.press(Key.left)
                time.sleep(0.3)
                keyboard.release(Key.left)
                return 'left'
            elif current_pos == 'right' :
                keyboard.press(Key.right)
                time.sleep(0.3)
                keyboard.release(Key.right)
                return 'right'

        # Detecting number on the trunk Level 2
        detected_number = self.trunk_number_detection(frame, trunk_top_left, trunk_bottom_right)
        if detected_number != -1 :
            print(f"Number {detected_number} Detected !")
            for i in range(detected_number-1) :
                if current_pos == 'left' :
                    keyboard.press(Key.left)
                    time.sleep(0.15)
                    keyboard.release(Key.left)
                elif current_pos == 'right' :
                    keyboard.press(Key.right)
                    time.sleep(0.15)
                    keyboard.release(Key.right)
        # Power Boost detection Level 4
        branch = self.detect_branch(roi_frame)
        power_boost = self.green_power_boost_detection(roi_frame)
        if power_boost == 'right' and self.bottom_right != True and branch != 'right':
            print("Collecting power boost right !")
            keyboard.press(Key.right)
            time.sleep(0.2)
            keyboard.release(Key.right)
            return 'right'
        elif power_boost == 'left' and self.bottom_left != True and branch != 'left' :
            print("Collecting power boost left !")
            keyboard.press(Key.left)
            time.sleep(0.2)
            keyboard.release(Key.left)
            return 'left'
        # Level 1
        if branch == 'right' :
            keyboard.press(Key.left)
            time.sleep(0.14)
            keyboard.release(Key.left)
            self.bottom_right = True
            self.bottom_left = False
            return 'left'
        elif branch == 'left' :
            keyboard.press(Key.right)
            time.sleep(0.14)
            keyboard.release(Key.right)
            self.bottom_left = True
            self.bottom_right = False
            return 'right'
        else :
            if current_pos == 'left' :
                keyboard.press(Key.left)
                time.sleep(0.14)
                keyboard.release(Key.left)
                self.bottom_left = False
                self.bottom_right = False
                return 'left'
            elif current_pos == 'right' :
                keyboard.press(Key.right)
                time.sleep(0.14)
                keyboard.release(Key.right)
                self.bottom_left = False
                self.bottom_right = False
                return 'right'


if __name__ == "__main__" :
    game = KarateKidAutomation()
    game.capture_frames()