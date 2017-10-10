using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectTracker
{
    public static class Extensions
    {
        public static T[] GetRow<T>(this T[,] input2DArray, int row) where T : IComparable
        {
            var width = input2DArray.GetLength(0);
            var height = input2DArray.GetLength(1);

            if (row >= height)
                throw new IndexOutOfRangeException("Row Index Out of Range");
            // Ensures the row requested is within the range of the 2-d array


            var returnRow = new T[width];
            for (var i = 0; i < width; i++)
                returnRow[i] = input2DArray[i, row];

            return returnRow;
        }

        public static T[] GetColumn<T>(this T[,] input2DArray, int col) where T : IComparable
        {
            var width = input2DArray.GetLength(0);
            var height = input2DArray.GetLength(1);

            if (col >= width)
                throw new IndexOutOfRangeException("Row Index Out of Range");
            // Ensures the row requested is within the range of the 2-d array


            var returnRow = new T[height];
            for (var i = 0; i < height; i++)
                returnRow[i] = input2DArray[col, i];

            return returnRow;
        }
    }
}
