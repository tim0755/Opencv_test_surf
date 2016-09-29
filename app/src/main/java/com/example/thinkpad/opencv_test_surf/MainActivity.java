package com.example.thinkpad.opencv_test_surf;

import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "opencvTest";

    private static final String StoragePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).getAbsolutePath();
    private static final String DIR = "/1";
    private static final String fileInputObject = DIR + "/fileInputObject.jpg";
    private static final String fileInputObjectPart = DIR + "/fileInputObjectPart.jpg";
    private static final String fileInputScene = DIR + "/fileInputScene.jpg";
    private static final String fileOutputObject = DIR + "/fileOutputObject.jpg";
    private static final String fileOutputImage = DIR + "/fileOutputImage.jpg";
    private static final String fileOutputMatch = DIR + "/fileOutputMatch.jpg";
    private static final String fileOutputMatchNew = DIR + "/fileOutputMatchNew.jpg";
    private static final String fileOutput0 = DIR + "/fileOutput0.jpg";
    private static final String fileOutput = DIR + "/fileOutput.jpg";

    static {
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "opencv successfully loader!");
        } else {
            Log.d(TAG, "opencv not loader!");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        SURFDetector();
    }

    private void SURFDetector() {

        Log.d(TAG, "getAbsolutePath:" + StoragePath + fileInputObject);

        Mat objectImage = Imgcodecs.imread(StoragePath + fileInputObject);
        Mat sceneImage = Imgcodecs.imread(StoragePath + fileInputScene);

        /*rectangeFillMat(objectImage, objectImage.get(0, 0), 0, 0, 800, 2160);
        rectangeFillMat(objectImage, objectImage.get(0, 0), 3040, 0, 800, 2160);
        rectangeFillMat(sceneImage, sceneImage.get(0, 0), 1100, 0, 1640, 2160);*/
        //Imgcodecs.imwrite(StoragePath + fileInputObjectPart, objectImage);

        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.AKAZE);

        featureDetector.detect(objectImage, objectKeyPoints);
        Log.d(TAG, "getAbsolutePath:" + objectKeyPoints.toArray());

        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);
        Log.d(TAG, "Computing descriptors...");
        descriptorExtractor.compute(objectImage, objectKeyPoints, objectDescriptors);

        // Create the matrix for output image.
        Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(), 1);//1, >0 rgb format???
        Scalar newKeypointColor = new Scalar(255, 0, 0);

        Log.d(TAG, "Drawing key points on object image...");
        Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);
        Imgcodecs.imwrite(StoragePath + fileOutputImage, outputImage);

        // Match object image with the scene image
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
        Log.d(TAG, "Detecting key points in background image...");
        featureDetector.detect(sceneImage, sceneKeyPoints);
        descriptorExtractor.compute(sceneImage, sceneKeyPoints, sceneDescriptors);
        Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2, 1);
        Scalar matchestColor = new Scalar(0, 255, 0);

        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
        Log.d(TAG, "Matching object and scene images...");
        descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2);

        Log.d(TAG, "Calculating good match list..." + matches.size() + " " + matches);
        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();

        float nndrRatio = 0.5f;

        List<KeyPoint> objKeypointlist1 = objectKeyPoints.toList();
        List<KeyPoint> scnKeypointlist2 = sceneKeyPoints.toList();

        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                Point pt1 = objKeypointlist1.get(m1.queryIdx).pt;
                Point pt2 = scnKeypointlist2.get(m1.trainIdx).pt;
                Log.d(TAG, "match:(" + (int) pt1.x + "," + (int) pt1.y + ") to (" + (int) pt2.x + "," + (int) pt2.y + ")");
                goodMatchesList.addLast(m1);
            }
        }

        if (goodMatchesList.size() >= 7) {
            Log.d(TAG, "Object Found!!!" + goodMatchesList.size());

            List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
            List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

            LinkedList<Point> objectPoints = new LinkedList<>();
            LinkedList<Point> scenePoints = new LinkedList<>();

            for (int i = 0; i < goodMatchesList.size(); i++) {
                objectPoints.addLast(objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt);
                scenePoints.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
            }

            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);

            int ransacReprojThreshold = 3;
            Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, ransacReprojThreshold);

            LinkedList<DMatch> goodMatchesListNew = new LinkedList<DMatch>();

            {
                MatOfPoint2f objMatOfPoint2fNew = new MatOfPoint2f();
                Core.perspectiveTransform(objMatOfPoint2f, objMatOfPoint2fNew, homography);
                Point[] objectPointsNew = objMatOfPoint2fNew.toArray();

                for (int i = 0; i < goodMatchesList.size(); i++) {
                    LinkedList<Point> scenePointsNew = new LinkedList<>();
                    MatOfPoint2f temp1 = new MatOfPoint2f();
                    MatOfPoint2f temp2 = new MatOfPoint2f();

                    temp1.fromArray(objectPointsNew[i]);
                    scenePointsNew.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
                    temp2.fromList(scenePointsNew);

                    Log.e(TAG, "Core.norm:" + Core.norm(temp1, temp2));
                    if (Core.norm(temp1, temp2) < ransacReprojThreshold) {
                        goodMatchesListNew.add(goodMatchesList.get(i));
                    }
                }
                Log.e(TAG, "size:" + goodMatchesList.size() + " new size:" + goodMatchesListNew.size());
            }

            Log.d(TAG, "homography:" + homography);
            Log.d(TAG, "" + homography.dump());

            Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

            obj_corners.put(0, 0, new double[]{0, 0});
            obj_corners.put(1, 0, new double[]{objectImage.cols(), 0});
            obj_corners.put(2, 0, new double[]{objectImage.cols(), objectImage.rows()});
            obj_corners.put(3, 0, new double[]{0, objectImage.rows()});

            {
                Mat img1 = Imgcodecs.imread(StoragePath + fileInputObject);
                Imgproc.line(img1, new Point(obj_corners.get(0, 0)), new Point(obj_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
                Imgproc.line(img1, new Point(obj_corners.get(1, 0)), new Point(obj_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
                Imgproc.line(img1, new Point(obj_corners.get(2, 0)), new Point(obj_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
                Imgproc.line(img1, new Point(obj_corners.get(3, 0)), new Point(obj_corners.get(0, 0)), new Scalar(0, 255, 0), 4);
                Imgcodecs.imwrite(StoragePath + fileOutput0, img1);
            }

            Log.d(TAG, "Transforming object corners to scene corners...");
            Core.perspectiveTransform(obj_corners, scene_corners, homography);

            Mat img = Imgcodecs.imread(StoragePath + fileInputScene);

            Imgproc.line(img, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(img, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(img, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
            Imgproc.line(img, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), new Scalar(0, 255, 0), 4);

            Log.d(TAG, "Drawing matches image...");
            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);

            Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage, sceneKeyPoints, goodMatches, matchoutput, matchestColor, newKeypointColor, new MatOfByte(), 2);

            Imgcodecs.imwrite(StoragePath + fileOutputMatch, matchoutput);
            {
                MatOfDMatch goodMatchesNew = new MatOfDMatch();
                goodMatchesNew.fromList(goodMatchesListNew);
                Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage, sceneKeyPoints, goodMatchesNew, matchoutput, matchestColor, newKeypointColor, new MatOfByte(), 2);
                Imgcodecs.imwrite(StoragePath + fileOutputMatchNew, matchoutput);
            }
            Imgcodecs.imwrite(StoragePath + fileOutput, img);
        } else {
            Log.e(TAG, "Object Not Found match number just only:" + goodMatchesList.size());
        }
    }

    private void rectangeFillMat(Mat src, double[] target, int x, int y, int w, int h) {

        long time = System.currentTimeMillis();
        Log.d(TAG, "rectangeFillMat!");
        if (x + w > src.cols() || y + h > src.rows()) {
            Log.d(TAG, "error parameters!");
            return;
        }

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                src.put(y + i, x + j, target);
            }
        }
        Log.d(TAG, "rectangeFillMat finish! use time:" + (System.currentTimeMillis() - time));
    }

}
