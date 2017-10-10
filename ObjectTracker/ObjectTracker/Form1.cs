using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.UI;
using System.Collections;

namespace ObjectTracker
{
    public partial class Form1 : Form
    {
        static VideoCapture _camera;

        public Form1()
        {
            InitializeComponent();

            //string captureSource = @"C:\Users\zumberc\Documents\Python Scripts\object_tracking_example.mp4"; // or live capture
            //string captureSource = @"C:\Users\zumberc\Documents\Python Scripts\Video.MOV";
            string captureSource = @"Video.MOV";

            try
            {
                _camera = new VideoCapture(captureSource);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex);
            }

            // Occurs when the application finishes processing and is about to enter the idle state
            Application.Idle += Application_Idle;
        }

        private void Application_Idle(object sender, EventArgs e)
        {
            Matrix<int> greenLower = new Matrix<int>(new int[] { 29, 86, 6 });
            Matrix<int> greenUpper = new Matrix<int>(new int[] { 64, 255, 255 });

            // grab the current frame
            Mat frame = _camera.QueryFrame();

            if (frame != null)
            {
                // resize the frame
                //CvInvoke.Resize(frame, frame, new Size(600, 600));

                Mat trackingFrame = new Mat();
                frame.CopyTo(trackingFrame);

                // blur the image to smooth it and reduce high frequency noise
                Mat blurred = new Mat();
                CvInvoke.GaussianBlur(frame, blurred, new Size(11, 11), 0);

                // convert to HSV color space since we defined the green range in the HSV color space
                Mat hsv = new Mat();
                CvInvoke.CvtColor(frame, hsv, ColorConversion.Bgr2Hsv);

                // construct a mask for the color "green", then perform a series of dilations and erotions to remove
                // any small blobs left in the mask
                Mat mask = new Mat();
                CvInvoke.InRange(hsv, greenLower, greenUpper, mask);
                // null, point(-1, 1), bordertype.constant, and morphologydefaultbordervalue are the default values according to documentation
                // http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=erode#erode
                CvInvoke.Erode(mask, mask, null, new Point(-1, -1), 2, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);
                // 
                // http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#dilate
                CvInvoke.Dilate(mask, mask, null, new Point(-1, -1), 2, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

                // find countours in the mask and initialize teh current (x, y) center of the ball
                Mat contoursMask = new Mat();
                mask.CopyTo(contoursMask);


                //http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#cv2.findContours
                VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(contoursMask, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

                // only proceed if at least one contour was found
                if (contours.Size > 0)
                {
                    TrackMultipleSingleObjects_WithMemory(trackingFrame, contours);
                    //TrackMultipleSingleObjects_NoMemory(trackingFrame, contours);
                    //TrackSingleLargestContour(trackingFrame, contours);
                }

                imageboxCamera.Image = frame;
                imageboxMask.Image = mask;
                //imageboxMask.Image = _matchImg;
                imageboxTracking.Image = trackingFrame;
            }
            else
            {
                // To show nothing when it's done
                imageboxCamera.Image = null;
                imageboxMask.Image = null;
                imageboxTracking.Image = null;
            }
        }

        private void TrackMultipleSingleObjects_WithMemory(Mat trackingFrame, VectorOfVectorOfPoint contours)
        {
            // Reduce all the found contours to a set of actual objects to process
            //Dictionary<string, TrackableObject> observedObjectsDict = ReduceContoursToObservedObjects(trackingFrame, contours);
            List<TrackableObject> observedObjectsList = ReduceContoursToObservedObjects(trackingFrame, contours);

            Dictionary<string, string> matchDict = new Dictionary<string, string>();

            // Try to match the processableObjects with the existing objects to create object permanance
            if (_objectsInScene.Count > 0)
            {
                // Create a feature match/distance matrix
                Tuple<bool, float, TrackableObject>[,] matchDistanceMatrix = new Tuple<bool, float, TrackableObject>[_objectsInScene.Count, observedObjectsList.Count];

                for (int x = 0; x < _objectsInScene.Count; x++)
                {
                    TrackableObject sceneObject = _objectsInScene[x];

                    for (int y = 0; y < observedObjectsList.Count; y++)
                    {
                        TrackableObject observedObject = observedObjectsList[y];

                        bool matched = false;
                        Mat homography = null; VectorOfKeyPoint modelKeyPoints; VectorOfKeyPoint observedKeyPoints; long matchTime; Mat mask;
                        using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
                        {
                            try
                            {
                                FindMatch(sceneObject.LastFrame, observedObject.LastFrame, out matchTime, out modelKeyPoints, out observedKeyPoints, matches, out mask, out homography);
                                //_matchImg = DrawMatches(sceneObject.LastFrame, observedObject.LastFrame, out matchTime);
                            }
                            catch (Exception ex)
                            {
                                System.Diagnostics.Debug.WriteLine(ex);
                            }

                            if (homography != null)
                                matched = true;
                        }

                        float distance = (float)Math.Abs(Math.Pow(sceneObject.FrameCenter.X - observedObject.FrameCenter.X, 2) +
                                    Math.Pow(sceneObject.FrameCenter.Y - observedObject.FrameCenter.Y, 2));

                        Tuple<bool, float, TrackableObject> data = new Tuple<bool, float, TrackableObject>(matched, distance, observedObject);

                        matchDistanceMatrix[x, y] = data;
                    }

                    var column = matchDistanceMatrix.GetColumn(x);
                    TrackableObject bestMatch = column.Where(r => r != null && r.Item1 == true).OrderBy(r => r.Item2).FirstOrDefault()?.Item3;

                    // minor smoothing
                    if (bestMatch == null)
                    {
                        bestMatch = column.Where(r => r != null && r.Item2 < 60).FirstOrDefault()?.Item3;
                    }

                    if (bestMatch != null)
                    {
                        matchDict.Add(sceneObject.ObjectIdentifier, bestMatch.ObjectIdentifier);
                    }
                }

                foreach (var matchItem in matchDict)
                {
                    TrackableObject sceneObj = _objectsInScene.First(x => x.ObjectIdentifier.Equals(matchItem.Key));
                    TrackableObject observedObj = observedObjectsList.First(x => x.ObjectIdentifier.Equals(matchItem.Value));

                    sceneObj.TrailPoints.Enqueue(observedObj.FrameCenter);
                    while (sceneObj.TrailPoints.Count > _buffer)
                        sceneObj.TrailPoints.Dequeue();

                    DrawTrackingTrail(trackingFrame, sceneObj);

                    sceneObj.FrameCenter = observedObj.FrameCenter;
                    sceneObj.FrameRadius = observedObj.FrameRadius;
                    sceneObj.BoundingRectangle = observedObj.BoundingRectangle;
                    sceneObj.LastFrame = observedObj.LastFrame;
                }
            }
            // Deal with collisions in the match dict?

            List<string> matchedIdentifiers = matchDict.Select(x => x.Key).ToList();
            // Iterate all the objects that were in the scene before and aren't matched now, remove them from the scene
            List<TrackableObject> objectsToRemove = _objectsInScene.Where(x => !matchedIdentifiers.Contains(x.ObjectIdentifier)).ToList();
            foreach (TrackableObject obj in objectsToRemove)
            {
                _objectsInScene.Remove(obj);
            }

            matchedIdentifiers = matchDict.Select(x => x.Value).ToList();
            // Iterate all the new objects that were not matched to an existing, and add them to the scene
            foreach (TrackableObject obj in observedObjectsList.Where(x => !matchedIdentifiers.Contains(x.ObjectIdentifier)))
            {
                _objectsInScene.Add(obj);
                _lifetimeObjects++;
            }

            foreach (TrackableObject obj in _objectsInScene)
            {
                CvInvoke.Rectangle(trackingFrame, obj.BoundingRectangle, new MCvScalar(255, 0, 0), 2, LineType.EightConnected, 0);
            }

            CvInvoke.PutText(trackingFrame, $"Scene: {_objectsInScene.Count.ToString()}", new Point(10, 30), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.65, new MCvScalar(0, 0, 255), 2);
            CvInvoke.PutText(trackingFrame, $"Total: {_lifetimeObjects.ToString()}", new Point(10, 60), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.65, new MCvScalar(0, 0, 255), 2);
        }

        Mat _matchImg { get; set; } = null;

        //private static Dictionary<string, TrackableObject> ReduceContoursToObservedObjects(Mat trackingFrame, VectorOfVectorOfPoint contours)
        private static List<TrackableObject> ReduceContoursToObservedObjects(Mat trackingFrame, VectorOfVectorOfPoint contours)
        {
            Dictionary<int, TrackableObject> observedObjectsDict = new Dictionary<int, TrackableObject>();

            // Get bounding rectangles for all contours, and extract the Mat from rects as the last seen frame for this object
            for (int i = 0; i < contours.Size; i++)
            {
                TrackableObject obj = new TrackableObject();
                IInputArray contour = contours[i];

                // enclosing circle
                CircleF minEnclosingCircle = CvInvoke.MinEnclosingCircle(contour);

                MCvMoments moments = CvInvoke.Moments(contour);
                Point center = new Point((int)(moments.M10 / moments.M00), (int)(moments.M01 / moments.M00));

                obj.FrameCenter = center;
                obj.FrameRadius = (int)minEnclosingCircle.Radius;
                obj.EnclosingCircle = minEnclosingCircle;
                // bounding rect
                VectorOfPoint contour_poly = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, contour_poly, 3, true);
                obj.BoundingRectangle = CvInvoke.BoundingRectangle(contour_poly);

                obj.LastFrame = new Mat(trackingFrame, obj.BoundingRectangle);

                observedObjectsDict.Add(i, obj);
            }
            // Get cirlces and centers and do inside calculations
            //Dictionary<string, TrackableObject> processableObjectsDict = new Dictionary<string, TrackableObject>();
            List<TrackableObject> processableObjectsList = new List<TrackableObject>();
            for (int i = 0; i < contours.Size; i++)
            {
                CircleF contourCircle = observedObjectsDict[i].EnclosingCircle;
                if (observedObjectsDict.Any(c => c.Key != i && IsMostlyInsideCircle(contourCircle, c.Value.EnclosingCircle)))
                {
                    // is inside, probably noise
                }
                else
                {
                    if (observedObjectsDict[i].FrameRadius > 10)
                    {
                        //processableObjectsDict.Add(observedObjectsDict[i].ObjectIdentifier, observedObjectsDict[i]);
                        processableObjectsList.Add(observedObjectsDict[i]);
                    }
                }
            }

            return processableObjectsList;
        }

        private static void DrawTrackingTrail(Mat trackingFrame, TrackableObject updatableSceneObj)
        {
            //loop over set of tracking points
            for (int i = 1; i < updatableSceneObj.TrailPoints.Count; i++)
            {
                if (updatableSceneObj.TrailPoints.ElementAt(i - 1) == null || updatableSceneObj.TrailPoints.ElementAt(i) == null)
                    continue;

                // check to see if enough points have been accumulated
                if ((updatableSceneObj.TrailPoints.Count >= 10) && (i == 1) && (updatableSceneObj.TrailPoints.ElementAt(updatableSceneObj.TrailPoints.Count - 10) != null))
                {
                    // compute the difference between x and y coordinate
                    // and re-initizalize teh direciton text variables
                    updatableSceneObj.dX = updatableSceneObj.TrailPoints.ElementAt(updatableSceneObj.TrailPoints.Count - 10).X - updatableSceneObj.TrailPoints.ElementAt(i).X;
                    updatableSceneObj.dY = updatableSceneObj.TrailPoints.ElementAt(updatableSceneObj.TrailPoints.Count - 10).Y - updatableSceneObj.TrailPoints.ElementAt(i).Y;
                    updatableSceneObj.dirX = String.Empty;
                    updatableSceneObj.dirY = String.Empty;

                    // ensure there is significant movement in the x-direction
                    if (Math.Abs(updatableSceneObj.dX) > 20)
                        updatableSceneObj.dirX = Math.Sign(updatableSceneObj.dX) == 1 ? "Right" : "Left";
                    // ensure there is significant movement in the y-direection
                    if (Math.Abs(updatableSceneObj.dY) > 20)
                        updatableSceneObj.dirY = Math.Sign(updatableSceneObj.dY) == 1 ? "Down" : "Up";

                    // handle when both directions are non-empty
                    if (!String.IsNullOrEmpty(updatableSceneObj.dirX) && !String.IsNullOrEmpty(updatableSceneObj.dirY))
                        updatableSceneObj.Direction = $"{updatableSceneObj.dirY}-{updatableSceneObj.dirX}";
                    // otherwise, only one direction is non-empty
                    else
                        updatableSceneObj.Direction = String.IsNullOrEmpty(updatableSceneObj.dirY) ? updatableSceneObj.dirX : updatableSceneObj.dirY;
                }

                // compute the thickness of the line and draw the connecting lines
                int thickness = (int)(Math.Sqrt(_buffer / ((float)(i + 1))) * 2.5);
                CvInvoke.Line(trackingFrame, updatableSceneObj.TrailPoints.ElementAt(i - 1), updatableSceneObj.TrailPoints.ElementAt(i), new MCvScalar(0, 0, 255), thickness);
            }
        }

        private void TrackMultipleSingleObjects_NoMemory(Mat trackingFrame, VectorOfVectorOfPoint contours)
        {
            // Get and store the bounding circles for all the contours so we can figure out which contours are contained in larger ones
            Dictionary<int, CircleF> contourCircleDict = new Dictionary<int, CircleF>();

            Rectangle[] boundRect = new Rectangle[contours.Size];
            VectorOfVectorOfPoint contours_poly = new VectorOfVectorOfPoint(contours.Size);

            for (int i = 0; i < contours.Size; i++)
            {
                IInputArray contour = contours[i];

                // enclosing circle
                CircleF minEnclosingCircle = CvInvoke.MinEnclosingCircle(contour);

                float x = minEnclosingCircle.Center.X;
                float y = minEnclosingCircle.Center.Y;
                float radius = minEnclosingCircle.Radius;

                MCvMoments moments = CvInvoke.Moments(contour);
                Point center = new Point((int)(moments.M10 / moments.M00), (int)(moments.M01 / moments.M00));

                CircleF circle = new CircleF(center, radius);

                contourCircleDict.Add(i, circle);

                // bonding rectangle
                VectorOfPoint vp = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, vp, 3, true);
                boundRect[i] = CvInvoke.BoundingRectangle(vp);
                //CvInvoke.ApproxPolyDP(contour, contours_poly[i], 3, true);
                //boundRect[i] = CvInvoke.BoundingRectangle(contours_poly[i]);
            }

            _state.ObjectsInView = 0;
            for (int i = 0; i < contours.Size; i++)
            {
                IInputArray contour = contours[i];
                CircleF contourCircle = contourCircleDict[i];

                // If the contour is contained in another contour, paint it red, else, blue
                MCvScalar colorScalar;
                if (contourCircleDict.Any(c => c.Key != i && Form1.IsMostlyInsideCircle(contourCircle, c.Value)))
                {
                    // red.. inner one.. ignore right now, probably noise?
                    colorScalar = new MCvScalar(0, 0, 255);
                }
                else
                {
                    // blue
                    colorScalar = new MCvScalar(255, 0, 0);
                    _state.ObjectsInView++;
                }

                Point center = new Point((int)contourCircle.Center.X, (int)contourCircle.Center.Y);

                if (contourCircle.Radius > 10)
                {
                    CvInvoke.Circle(trackingFrame, center, (int)contourCircle.Radius,
                       colorScalar, 2);
                    CvInvoke.Circle(trackingFrame, center, 5, new MCvScalar(0, 0, 255), -1);
                    CvInvoke.Rectangle(trackingFrame, boundRect[i], colorScalar, 2, LineType.EightConnected, 0);
                }
            }

            CvInvoke.PutText(trackingFrame, _state.ObjectsInView.ToString(), new Point(10, trackingFrame.Height - 10), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.65, new MCvScalar(0, 0, 255), 2);
        }

        private void TrackSingleLargestContour(Mat trackingFrame, VectorOfVectorOfPoint contours)
        {
            IInputArray largestContour = null;
            double largestContourArea = double.MinValue;
            for (int i = 0; i < contours.Size; i++)
            {
                IInputArray contour = contours[i];
                double area = CvInvoke.ContourArea(contour);
                if (area > largestContourArea)
                {
                    largestContour = contour;
                    largestContourArea = area;
                }
            }

            if (largestContour != null)
            {
                CircleF minEnclosingCircle = CvInvoke.MinEnclosingCircle(largestContour);
                float x = minEnclosingCircle.Center.X;
                float y = minEnclosingCircle.Center.Y;
                float radius = minEnclosingCircle.Radius;

                MCvMoments moments = CvInvoke.Moments(largestContour);
                Point center = new Point((int)(moments.M10 / moments.M00), (int)(moments.M01 / moments.M00));

                _obj.FrameCenter = center;
                _obj.FrameRadius = (int)radius;

                // only proceed if the radius meets a minimum size
                //if (radius > 10)
                if (_obj != null && _obj.FrameRadius > 10)
                {
                    // draw the circle and the centroid on the frame
                    CvInvoke.Circle(trackingFrame, _obj.FrameCenter, _obj.FrameRadius,
                       new MCvScalar(0, 255, 255), 2);
                    CvInvoke.Circle(trackingFrame, _obj.FrameCenter, 5, new MCvScalar(0, 0, 255), -1);

                    _obj.TrailPoints.Enqueue(center);
                    while (_obj.TrailPoints.Count > _buffer)
                        _obj.TrailPoints.Dequeue();
                }
            }

            //loop over set of tracking points
            for (int i = 1; i < _obj.TrailPoints.Count; i++)
            {
                if (_obj.TrailPoints.ElementAt(i - 1) == null || _obj.TrailPoints.ElementAt(i) == null)
                    continue;

                // check to see if enough points have been accumulated
                if ((_obj.TrailPoints.Count >= 10) && (i == 1) && (_obj.TrailPoints.ElementAt(_obj.TrailPoints.Count - 10) != null))
                {
                    // compute the difference between x and y coordinate
                    // and re-initizalize teh direciton text variables
                    _obj.dX = _obj.TrailPoints.ElementAt(_obj.TrailPoints.Count - 10).X - _obj.TrailPoints.ElementAt(i).X;
                    _obj.dY = _obj.TrailPoints.ElementAt(_obj.TrailPoints.Count - 10).Y - _obj.TrailPoints.ElementAt(i).Y;
                    _obj.dirX = String.Empty;
                    _obj.dirY = String.Empty;

                    // ensure there is significant movement in the x-direction
                    if (Math.Abs(_obj.dX) > 20)
                        _obj.dirX = Math.Sign(_obj.dX) == 1 ? "Right" : "Left";
                    // ensure there is significant movement in the y-direection
                    if (Math.Abs(_obj.dY) > 20)
                        _obj.dirY = Math.Sign(_obj.dY) == 1 ? "Down" : "Up";

                    // handle when both directions are non-empty
                    if (!String.IsNullOrEmpty(_obj.dirX) && !String.IsNullOrEmpty(_obj.dirY))
                        _obj.Direction = $"{_obj.dirY}-{_obj.dirX}";
                    // otherwise, only one direction is non-empty
                    else
                        _obj.Direction = String.IsNullOrEmpty(_obj.dirY) ? _obj.dirX : _obj.dirY;
                }

                // compute the thickness of the line and draw the connecting lines
                int thickness = (int)(Math.Sqrt(_buffer / ((float)(i + 1))) * 2.5);
                CvInvoke.Line(trackingFrame, _obj.TrailPoints.ElementAt(i - 1), _obj.TrailPoints.ElementAt(i), new MCvScalar(0, 0, 255), thickness);
            }

            CvInvoke.PutText(trackingFrame, _obj.Direction, new Point(10, 30), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.65, new MCvScalar(0, 0, 255), 2);
            CvInvoke.PutText(trackingFrame, $"dx: {_obj.dX}, dy: {_obj.dY}", new Point(10, 60), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.35, new MCvScalar(0, 0, 255), 1);
        }

        /// <summary>
        /// Determines whether the cirlce is inside the parameter circle.
        /// </summary>
        /// <param name="c1">The circle to check if inside of.</param>
        /// <returns>
        ///   <c>true</c> if [is inside circle]; otherwise, <c>false</c>.
        /// </returns>
        private static bool IsInsideCircle(CircleF c2, CircleF c1)
        {
            float x1 = c1.Center.X; float y1 = c1.Center.Y; float r1 = c1.Radius;
            float x2 = c2.Center.X; float y2 = c2.Center.Y; float r2 = c2.Radius;

            var d = Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2));

            if (r1 > (d + r2))
                return true;
            else
                return false;
        }

        private static bool IsMostlyInsideCircle(CircleF c2, CircleF c1)
        {
            if (c2.Radius > c1.Radius)
                return false;

            float x1 = c1.Center.X; float y1 = c1.Center.Y; float r1 = c1.Radius;
            float x2 = c2.Center.X; float y2 = c2.Center.Y; float r2 = c2.Radius;

            var d = Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2));

            if (r1 > d)
                return true;
            else
                return false;
        }

        private static void FindMatch(Mat modelImage, Mat observedImage, out long matchTime, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int k = 2;
            double uniquenessThreshold = 0.80;

            Stopwatch watch;
            homography = null;

            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();

            using (UMat uModelImage = modelImage.GetUMat(AccessType.Read))
            using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))
            {
                KAZE featureDetector = new KAZE();

                //extract features from the object image
                Mat modelDescriptors = new Mat();
                featureDetector.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);

                watch = Stopwatch.StartNew();

                // extract features from the observed image
                Mat observedDescriptors = new Mat();
                featureDetector.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);

                // Bruteforce, slower but more accurate
                // You can use KDTree for faster matching with slight loss in accuracy
                using (Emgu.CV.Flann.LinearIndexParams ip = new Emgu.CV.Flann.LinearIndexParams())
                using (Emgu.CV.Flann.SearchParams sp = new SearchParams())
                using (DescriptorMatcher matcher = new FlannBasedMatcher(ip, sp))
                {
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                            matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                                observedKeyPoints, matches, mask, 2);
                    }
                }
                watch.Stop();

            }
            matchTime = watch.ElapsedMilliseconds;
        }

        public static Mat DrawMatches(Mat modelImage, Mat observedImage, out long matchTime)
        {
            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelImage, observedImage, out matchTime, out modelKeyPoints, out observedKeyPoints, matches,
                   out mask, out homography);

                //Draw the matched keypoints
                Mat result = new Mat();

                //foreach (var a in matches.ToArrayOfArray())
                //{
                //    float minDist = a.Min(x => x.Distance);
                //    float maxDist = a.Max(x => x.Distance);
                //    System.Diagnostics.Debug.WriteLine($"{minDist} : {maxDist}");
                //}

                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                   matches, result,
                   //new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), 
                   new MCvScalar(0, 255, 0), new MCvScalar(0, 0, 255),
                   mask);

                #region draw the projected region on the image

                if (homography != null)
                {
                    //draw a rectangle along the projected model
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    PointF[] pts = new PointF[]
                    {
                      new PointF(rect.Left, rect.Bottom),
                      new PointF(rect.Right, rect.Bottom),
                      new PointF(rect.Right, rect.Top),
                      new PointF(rect.Left, rect.Top)
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);

#if NETFX_CORE
               Point[] points = Extensions.ConvertAll<PointF, Point>(pts, Point.Round);
#else
                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
#endif
                    using (VectorOfPoint vp = new VectorOfPoint(points))
                    {
                        CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                    }
                }
                #endregion

                return result;

            }
        }

        const int _buffer = 32;
        TrackableObject _obj = new TrackableObject();
        ApplicationState _state = new ApplicationState();

        List<TrackableObject> _objectsInScene = new List<TrackableObject>();
        int _lifetimeObjects = 0;

        class ApplicationState
        {
            public int ObjectsInView { get; set; } = 0;
            public int LifetimeObjects { get; set; } = 0;

        }

        class TrackableObject
        {
            public string ObjectIdentifier { get; } = Guid.NewGuid().ToString();

            //public List<Point> TrackingPoints { get; } = new List<Point>();
            public Queue<Point> TrailPoints { get; } = new Queue<Point>(_buffer);

            public Rectangle BoundingRectangle { get; set; }
            public CircleF EnclosingCircle { get; set; }
            public Point FrameCenter { get; set; }
            public int FrameRadius { get; set; }

            public Mat LastFrame { get; set; }

            public float dX { get; set; } = 0.0f;
            public float dY { get; set; } = 0.0f;
            public string dirX { get; set; } = String.Empty;
            public string dirY { get; set; } = String.Empty;
            public string Direction = String.Empty;
        }
    }
}
