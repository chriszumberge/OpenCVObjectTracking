using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Linq;

namespace ObjectTracker
{
    public partial class Form1 : Form
    {
        static VideoCapture _camera;

        public Form1()
        {
            InitializeComponent();

            //string captureSource = @"C:\Users\zumberc\Documents\Python Scripts\object_tracking_example.mp4"; // or live capture
            string captureSource = @"C:\Users\zumberc\Documents\Python Scripts\Video.MOV";

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

                    TrackMultipleSingleObjects_NoMemory(trackingFrame, contours);
                    //TrackSingleLargestContour(trackingFrame, contours);
                }

                imageboxCamera.Image = frame;
                imageboxMask.Image = mask;
                imageboxTracking.Image = trackingFrame;
            }
            else
            {
                // To show nothing when it's done
                //imageboxCamera.Image = null;
                //imageboxMask.Image = null;
                //imageboxTracking.Image = null;
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
                CvInvoke.ApproxPolyDP(contour, contours_poly[i], 3, true);
                boundRect[i] = CvInvoke.BoundingRectangle(contours_poly[i]);
            }

            _state.ObjectsInView = 0;
            for (int i = 0; i < contours.Size; i++)
            {
                IInputArray contour = contours[i];
                CircleF contourCircle = contourCircleDict[i];

                // If the contour is contained in another contour, paint it red, else, blue
                MCvScalar colorScalar;
                if (contourCircleDict.Any(c => c.Key != i && Form1.IsInsideCircle(contourCircle, c.Value)))
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

                obj.FrameCenter = center;
                obj.FrameRadius = (int)radius;

                // only proceed if the radius meets a minimum size
                //if (radius > 10)
                if (obj != null && obj.FrameRadius > 10)
                {
                    // draw the circle and the centroid on the frame
                    CvInvoke.Circle(trackingFrame, obj.FrameCenter, obj.FrameRadius,
                       new MCvScalar(0, 255, 255), 2);
                    CvInvoke.Circle(trackingFrame, obj.FrameCenter, 5, new MCvScalar(0, 0, 255), -1);

                    obj.TrailPoints.Enqueue(center);
                    while (obj.TrailPoints.Count > _buffer)
                        obj.TrailPoints.Dequeue();
                }
            }

            //loop over set of tracking points
            for (int i = 1; i < obj.TrailPoints.Count; i++)
            {
                if (obj.TrailPoints.ElementAt(i - 1) == null || obj.TrailPoints.ElementAt(i) == null)
                    continue;

                // check to see if enough points have been accumulated
                if ((obj.TrailPoints.Count >= 10) && (i == 1) && (obj.TrailPoints.ElementAt(obj.TrailPoints.Count - 10) != null))
                {
                    // compute the difference between x and y coordinate
                    // and re-initizalize teh direciton text variables
                    obj.dX = obj.TrailPoints.ElementAt(obj.TrailPoints.Count - 10).X - obj.TrailPoints.ElementAt(i).X;
                    obj.dY = obj.TrailPoints.ElementAt(obj.TrailPoints.Count - 10).Y - obj.TrailPoints.ElementAt(i).Y;
                    obj.dirX = String.Empty;
                    obj.dirY = String.Empty;

                    // ensure there is significant movement in the x-direction
                    if (Math.Abs(obj.dX) > 20)
                        obj.dirX = Math.Sign(obj.dX) == 1 ? "Right" : "Left";
                    // ensure there is significant movement in the y-direection
                    if (Math.Abs(obj.dY) > 20)
                        obj.dirY = Math.Sign(obj.dY) == 1 ? "Down" : "Up";

                    // handle when both directions are non-empty
                    if (!String.IsNullOrEmpty(obj.dirX) && !String.IsNullOrEmpty(obj.dirY))
                        obj.Direction = $"{obj.dirY}-{obj.dirX}";
                    // otherwise, only one direction is non-empty
                    else
                        obj.Direction = String.IsNullOrEmpty(obj.dirY) ? obj.dirX : obj.dirY;
                }

                // compute the thickness of the line and draw the connecting lines
                int thickness = (int)(Math.Sqrt(_buffer / ((float)(i + 1))) * 2.5);
                CvInvoke.Line(trackingFrame, obj.TrailPoints.ElementAt(i - 1), obj.TrailPoints.ElementAt(i), new MCvScalar(0, 0, 255), thickness);
            }

            CvInvoke.PutText(trackingFrame, obj.Direction, new Point(10, 30), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.65, new MCvScalar(0, 0, 255), 2);
            CvInvoke.PutText(trackingFrame, $"dx: {obj.dX}, dy: {obj.dY}", new Point(10, 60), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.35, new MCvScalar(0, 0, 255), 1);
        }

        /// <summary>
        /// Determines whether the cirlce is inside the parameter circle.
        /// </summary>
        /// <param name="c1">The circle to check if inside of.</param>
        /// <returns>
        ///   <c>true</c> if [is inside circle]; otherwise, <c>false</c>.
        /// </returns>
        public static bool IsInsideCircle(CircleF c2, CircleF c1)
        {
            float x1 = c1.Center.X; float y1 = c1.Center.Y; float r1 = c1.Radius;
            float x2 = c2.Center.X; float y2 = c2.Center.Y; float r2 = c2.Radius;

            var d = Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2));

            if (r1 > (d + r2))
                return true;
            else
                return false;
        }

        const int _buffer = 32;
        TrackableObject obj = new TrackableObject();
        ApplicationState _state = new ApplicationState();

        class ApplicationState
        {
            public int ObjectsInView { get; set; } = 0;
            public int LifetimeObjects { get; set; } = 0;

        }

        class TrackableObject
        {
            public string ObjectIdentifier { get; } = Guid.NewGuid().ToString();

            public List<Point> TrackingPoints { get; } = new List<Point>();
            public Queue<Point> TrailPoints { get; } = new Queue<Point>(_buffer);
            public Point FrameCenter { get; set; }
            public int FrameRadius { get; set; }

            public float dX { get; set; } = 0.0f;
            public float dY { get; set; } = 0.0f;
            public string dirX { get; set; } = String.Empty;
            public string dirY { get; set; } = String.Empty;
            public string Direction = String.Empty;
        }
    }
}
