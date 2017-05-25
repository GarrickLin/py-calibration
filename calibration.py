import cv2
import numpy as np
import json
from utils import enum
import time
from collections import OrderedDict
import xmltodict
from image_generator import gen_images
import cPickle as pickle

Pattern = enum(
    "NOT_EXISTING",
    "CHESSBOARD",
    "CIRCLES_GRID",
    "ASYMMETRIC_CIRCLES_GRID")
InputType = enum("INVALID", "CAMERA", "VIDEO_FILE", "IMAGE_LIST", "RTSP_STREAM")
Action = enum(DETECTION=0, CAPTURING=1, CALIBRATED=2)


class Settings:
    def __init__(self):
        # [w, h]
        self.boardSize = [0, 0]
        self.goodInput = False
        self.inputType = InputType.INVALID

    def write(self, f):
        params_dict = {}
        params_dict["BoardSize_Width"] = self.boardSize[0]
        params_dict["BoardSize_Height"] = self.boardSize[1]
        params_dict["Square_Size"] = self.squareSize
        params_dict["Calibrate_Pattern"] = self.patternToUse
        params_dict["Calibrate_NrOfFrameToUse"] = self.nrFrames
        params_dict["Calibrate_FixAspectRatio"] = self.aspectRatio
        params_dict["Calibrate_AssumeZeroTangentialDistortion"] = self.calibZeroTangentDist
        params_dict["Calibrate_FixPrincipalPointAtTheCenter"] = self.calibFixPrincipalPoint
        params_dict["Write_DetectedFeaturePoints"] = self.writePoints
        params_dict["Write_extrinsicParameters"] = self.writeExtrinsics
        params_dict["Write_outputFileName"] = self.outputFileName
        params_dict["Show_UndistortedImage"] = self.showUndistorsed
        params_dict["Input_FlipAroundHorizontalAxis"] = self.flipVertical
        params_dict["Input_Delay"] = self.delay
        params_dict["Input"] = self.input

        json.dump(params_dict, f, indent=4)

    def read(self, node):
        self.boardSize[0] = int(node["BoardSize_Width"])
        self.boardSize[1] = int(node["BoardSize_Height"])
        self.boardSize = tuple(self.boardSize)
        self.patternToUse = node["Calibrate_Pattern"].strip('"')
        self.squareSize = float(node["Square_Size"])
        self.nrFrames = int(node["Calibrate_NrOfFrameToUse"])
        self.aspectRatio = float(node["Calibrate_FixAspectRatio"])
        self.writePoints = int(node["Write_DetectedFeaturePoints"])
        self.writeExtrinsics = int(node["Write_extrinsicParameters"])
        self.outputFileName = node["Write_outputFileName"].strip('"')
        self.calibZeroTangentDist = int(
            node["Calibrate_AssumeZeroTangentialDistortion"])
        self.calibFixPrincipalPoint = int(
            node["Calibrate_FixPrincipalPointAtTheCenter"])
        self.useFisheye = int(node["Calibrate_UseFisheyeModel"])
        self.flipVertical = int(node["Input_FlipAroundHorizontalAxis"])
        self.showUndistorsed = int(node["Show_UndistortedImage"])
        self.input = node["Input"].strip('"')
        self.delay = int(node["Input_Delay"])
        self.fixK1 = int(node["Fix_K1"])
        self.fixK2 = int(node["Fix_K2"])
        self.fixK3 = int(node["Fix_K3"])
        self.fixK4 = int(node["Fix_K4"])
        self.fixK5 = int(node["Fix_K5"])

        self.validate()

    def validate(self):
        self.goodInput = True
        if self.boardSize[0] <= 0 or self.boardSize[1] <= 0:
            print "Invalid Board size:", boardSize
            self.goodInput = False
        if self.squareSize < 10e-6:
            print "Invalid square size", squareSize
            self.goodInput = False
        if self.nrFrames <= 0:
            print "Invalid number of frames", self.nrFrames
            self.goodInput = False
        if not self.input:
            self.inputType = InputType.INVALID
        else:
            if self.input.isdigit():
                self.inputType = InputType.CAMERA
                self.cameraID = int(self.input)
            elif self.input.endswith((".xml", ".yml", ".json")):
                self.inputType = InputType.IMAGE_LIST
                self.imageList = self.readStringList(self.input)
            elif self.input.endswith((".avi", ".mp4", ".flv")):
                self.inputType = InputType.VIDEO_FILE
            elif self.input.startswith("rtsp"):
                self.inputType = InputType.RTSP_STREAM
            if self.inputType == InputType.CAMERA:
                self.inputCapture = cv2.VideoCapture(self.cameraID)
            if self.inputType == InputType.VIDEO_FILE:
                self.inputCapture = cv2.VideoCapture(self.input)
            if self.inputType != InputType.IMAGE_LIST and not self.inputCapture.isOpened():
                self.inputType = InputType.INVALID

        if self.inputType == InputType.INVALID:
            print "Input does not exist:", self.input
            self.goodInput = False

        self.flag = 0
        if self.calibFixPrincipalPoint:
            self.flag |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if self.calibZeroTangentDist:
            self.flag |= cv2.CALIB_ZERO_TANGENT_DIST
        if self.aspectRatio:
            self.flag |= cv2.CALIB_FIX_ASPECT_RATIO
        if self.fixK1:
            self.flag |= cv2.CALIB_FIX_K1
        if self.fixK2:
            self.flag |= cv2.CALIB_FIX_K2
        if self.fixK3:
            self.flag |= cv2.CALIB_FIX_K3
        if self.fixK4:
            self.flag |= cv2.CALIB_FIX_K4
        if self.fixK5:
            self.flag |= cv2.CALIB_FIX_K5

        if self.useFisheye:
            self.flag = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            if self.fixK1:
                self.flag |= cv2.fisheye.CALIB_FIX_K1
            if self.fixK2:
                self.flag |= cv2.fisheye.CALIB_FIX_K2
            if self.fixK3:
                self.flag |= cv2.fisheye.CALIB_FIX_K3
            if self.fixK4:
                self.flag |= cv2.fisheye.CALIB_FIX_K4

        self.calibrationPattern = Pattern.NOT_EXISTING
        if self.patternToUse == "CHESSBOARD":
            self.calibrationPattern = Pattern.CHESSBOARD
        elif self.patternToUse == "CIRCLES_GRID":
            self.calibrationPattern = Pattern.CIRCLES_GRID
        elif self.patternToUse == "ASYMMETRIC_CIRCLES_GRID":
            self.calibrationPattern = Pattern.ASYMMETRIC_CIRCLES_GRID
        if self.calibrationPattern == Pattern.NOT_EXISTING:
            print "Camera calibration mode does not exist:", self.patternToUse
            self.goodInput = False

        self.atImageList = 0

    def nextImage(self):
        if self.inputType == InputType.IMAGE_LIST:
            return gen_images(self.imageList)
        else:
            raise NotImplementedError

    def readStringList(self, filename):       
        if filename.endswith(".xml"):
            imlist = xmltodict.parse(open(filename))["opencv_storage"]["images"]            
            imlist = imlist.split("\n")
            return imlist
        elif filename.endswith(".json"):
            imlist = json.load(open(filename))["opencv_storage"]["images"]
            imlist = imlist.split("\n")
            return imlist            

def computeReprojectionErrors(
        objectPoints,
        imagePoints,
        rvecs,
        tvecs,
        cameraMatrix,
        distCoeffs,
        fisheye):
    imagePoints2 = []
    totalPoints = 0
    totalErr = 0
    perViewErrors = []
    for objectPoint, imagePoint, rvec, tvec in zip(
            objectPoints, imagePoints, rvecs, tvecs):
        if fisheye:
            imagePoints2, jacobian = cv2.fisheye.projectPoints(
                objectPoint, rvec, tvec, cameraMatrix, distCoeffs)
        else:
            imagePoints2, jacobian = cv2.projectPoints(
                objectPoint, rvec, tvec, cameraMatrix, distCoeffs)
        err = cv2.norm(imagePoint, imagePoints2, cv2.NORM_L2)
        n = len(objectPoint)
        perViewErrors.append(err / np.sqrt(n))
        totalErr += err * err
        totalPoints += n

    return np.sqrt(totalErr / totalPoints), perViewErrors


def calcBoardCornerPositions(
        boardSize,
        squareSize,
        patternType=Pattern.CHESSBOARD):
    if patternType == Pattern.CHESSBOARD:
        corners = np.zeros((np.prod(boardSize), 3), np.float32)
        corners[:, :2] = np.indices(boardSize).T.reshape(-1, 2)
        corners *= squareSize        
        return corners.reshape(-1, 1, 3)
    else:
        raise NotImplementedError


def runCalibration(s, imageSize, imagePoints):
    cameraMatrix = np.identity(3)
    # ! [fixed_aspect]
    if s.flag & cv2.CALIB_FIX_ASPECT_RATIO:
        cameraMatrix[0][0] = s.aspectRatio
    # ! [fixed_aspect]
    if s.useFisheye:
        distCoeffs = np.zeros(4)
    else:
        distCoeffs = np.zeros(8)

    pattern_points = calcBoardCornerPositions(
        s.boardSize, s.squareSize, s.calibrationPattern)
    objectPoints = [pattern_points] * len(imagePoints)

    if s.useFisheye:
        rms, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.fisheye.calibrate(
            objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, None, None, flags=s.flag)
    else:
        rms, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, None, None, s.flag)

    print "Re-projection error reported by calibrateCamera:", rms

    ok = cv2.checkRange(cameraMatrix) and cv2.checkRange(distCoeffs)

    totalAvgErr, reprojErrs = computeReprojectionErrors(
        objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, s.useFisheye)

    return ok, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, totalAvgErr


def saveCameraParams(
        s,
        imageSize,
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs,
        reprojErrs,
        imagePoints,
        totalAvgErr):
    params = OrderedDict()
    params["calibration_time"] = time.ctime()
    params["nr_of_frames"] = max(len(rvecs), len(reprojErrs))
    params["image_width"] = imageSize[0]
    params["image_height"] = imageSize[1]
    params["board_width"] = s.boardSize[0]
    params["board_height"] = s.boardSize[1]
    params["square_size"] = s.squareSize

    if s.flag & cv2.CALIB_FIX_ASPECT_RATIO:
        params["fix_aspect_ratio"] = s.aspectRatio

    params["flags"] = s.flag
    params["fisheye_model"] = s.useFisheye
    params["camera_matrix"] = cameraMatrix.tolist()
    params["distortion_coefficients"] = distCoeffs.tolist()
    params["avg_reprojection_error"] = totalAvgErr
    params["per_view_reprojection_errors"] = np.array(reprojErrs).tolist()

    json.dump(params, open(s.outputFileName, "w"), indent=4)


def runCalibrationAndSave(s, imageSize, imagePoints):
    ok, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, totalAvgErr = runCalibration(
        s, imageSize, imagePoints)
    if ok:
        print "Calibration succeeded"
    else:
        print "Calibration failed"
    print ". avg re projection error =", totalAvgErr

    if ok:
        saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
                         reprojErrs, imagePoints,
                         totalAvgErr)

    return ok, cameraMatrix, distCoeffs


def main(inputSettingsFile="data/default.json"):
    s = Settings()   
    if inputSettingsFile.endswith(".json"):
        fs = json.load(open(inputSettingsFile))["opencv_storage"]["Settings"]
    s.read(fs)
    if not s.goodInput:
        print "Invalid input detected. Application stopping."
        return -1
    mode = Action.CAPTURING if s.inputType == InputType.IMAGE_LIST else Action.DETECTION
    prevTimestamp = 0
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    imagePoints = []
    cameraMatrix = None
    distCoeffs = None
    imageSize = None

    # ! [get_input]
    for view in s.nextImage():
        blinkOutput = False

        # -----  If no more image, or got enough, then stop calibration and sho
        if mode == Action.CAPTURING and len(imagePoints) >= s.nrFrames:
            ok, cameraMatrix, distCoeffs = runCalibrationAndSave(s, imageSize, 
                                                                imagePoints)
            if ok:
                mode = Action.CALIBRATED
            else:
                mode = Action.DETECTION
        # If there are no more images stop the loop
        if view is None:
            # if calibration threshold was not reached yet, calibrate now
            if mode != Action.CALIBRATED and len(imagePoints) > 0:
                ok, cameraMatrix, distCoeffs = runCalibrationAndSave(s, imageSize, imagePoints)
            break

        # ! [get_input]

        imageSize = view.shape[:2][::-1]
        if s.flipVertical:
            view = cv2.flip(view, 0)

        # ! [find_pattern]
        chessBoardFlags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        if not s.useFisheye:
            chessBoardFlags |= cv2.CALIB_CB_FAST_CHECK

        if s.calibrationPattern == Pattern.CHESSBOARD:
            found, pointBuf = cv2.findChessboardCorners(
                view, s.boardSize, chessBoardFlags)
        else:
            found = False
        # ! [find_pattern]

        # ! [pattern_found]
        # If done with success
        if found:
            # improve the found corners' coordinate accuracy for chessboard
            if s.calibrationPattern == Pattern.CHESSBOARD:
                viewGray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(viewGray, pointBuf, (11, 11), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1))

            ## For camera only take new samples after delay time
            #if mode == Action.CAPTURING and (
                    #not s.inputCapture.isOpened() or time.time() -
                    #prevTimestamp > s.delay):
                #prevTimestamp = time.time()
            if mode == Action.CAPTURING:
                imagePoints.append(pointBuf)

            cv2.drawChessboardCorners(view, s.boardSize, pointBuf, found)
        # ! [pattern_found]

        # ------------------------------------- Output Text -------------------
        # ! [output_text]
        msg = "100/100" if mode == Action.CAPTURING else "Calibrated" if mode == Action.CALIBRATED else "Press 'g' to start"
        textSize, baseLine = cv2.getTextSize(msg, 1, 1, 1)
        rows, cols = imageSize
        textOrigin = (cols - 2 * textSize[0] - 10, rows - 2 * baseLine - 10)

        if mode == Action.CAPTURING:
            if s.showUndistorsed:
                msg = "%d/%d Undist" % (len(imagePoints), s.nrFrames)
            else:
                msg = "%d/%d" % (len(imagePoints), s.nrFrames)
        cv2.putText(
            view,
            msg,
            textOrigin,
            1,
            1,
            GREEN if mode == Action.CALIBRATED else RED)

        if blinkOutput:
            view = cv2.bitwise_not(view)
        # --------------------------- Video capture  output  undistorted ------
        # ! [output_undistorted]
        if mode == Action.CALIBRATED and s.showUndistorsed:
            temp = view.copy()
            if s.useFisheye:
                view = cv2.fisheye.undistortImage(
                    temp, cameraMatrix, distCoeffs)
            else:
                view = cv2.undistort(temp, cameraMatrix, distCoeffs)
        # ! [output_undistorted]
        # ------------------------------ Show image and check for input command
        # ! [await_input]
        cv2.imshow("Image View", view)
        #key = cv2.waitKey(50 if s.inputCapture.isOpened() else s.delay)
        key = cv2.waitKey(50)

        if key == 27:
            break
        if key == ord('u') and mode == Action.CALIBRATED:
            s.showUndistorsed = not s.showUndistorsed

        #if s.inputCapture.isOpened() and key == ord('g'):
            #mode = Action.CAPTURING
            #imagePoints = []
            
        # ! [await_input]

    # -----------------------Show the undistorted image for the image list
    # ! [show_results]
    print "cameraMatrix\n", cameraMatrix, "\n"
    print "distCoeffs\n", distCoeffs, "\n"
    if s.inputType == InputType.IMAGE_LIST and s.showUndistorsed:
        if s.useFisheye:
            newCamMat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                cameraMatrix, distCoeffs, imageSize, np.identity(3), None, 1, imageSize, 0.55)
            #print imageSize
            #print np.identity(3)
            #print newCamMat
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                cameraMatrix, distCoeffs, np.identity(3), newCamMat, imageSize, cv2.CV_16SC2)
        else:
            newCamMat, roi = cv2.getOptimalNewCameraMatrix(
                cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0)
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix, distCoeffs, np.identity(3), newCamMat, imageSize, cv2.CV_16SC2)
            
        pickle.dump((map1, map2), open("data/map.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        for i, imgname in enumerate(s.imageList):
            view = cv2.imread(imgname)
            if view is None:
                continue
            rview = cv2.remap(view, map1, map2, cv2.INTER_LINEAR)
            cv2.imshow("Image View", rview)
            c = cv2.waitKey(0)
            if c == 27:
                break
    # ! [show_results]


if __name__ == "__main__":
    main()
