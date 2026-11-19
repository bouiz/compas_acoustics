using System;
using System.Globalization;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Python's ``format(x, ".{p}g")``, so that panels print numbers as compas_acoustics does.</summary>
    public static class PythonFormat
    {
        public static string G(double value, int precision = 6)
        {
            if (double.IsNaN(value))
                return "nan";
            if (double.IsInfinity(value))
                return value > 0 ? "inf" : "-inf";
            if (value == 0)
                return "0";
            precision = Math.Max(precision, 1);
            // the mantissa and exponent after rounding to `precision` significant digits
            var e = value.ToString("E" + (precision - 1), CultureInfo.InvariantCulture);
            var mark = e.IndexOf('E');
            var exponent = int.Parse(e.Substring(mark + 1), CultureInfo.InvariantCulture);
            if (exponent < -4 || exponent >= precision)
            {
                var mantissa = Trim(e.Substring(0, mark));
                return mantissa + "e" + (exponent < 0 ? "-" : "+") + Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture);
            }
            var rounded = double.Parse(e, CultureInfo.InvariantCulture);
            return Trim(rounded.ToString("F" + Math.Max(0, precision - 1 - exponent), CultureInfo.InvariantCulture));
        }

        /// <summary>Python's ``str`` of a float: the shortest text that reads back as the value.</summary>
        public static string Repr(double value)
        {
            if (double.IsNaN(value))
                return "nan";
            if (double.IsInfinity(value))
                return value > 0 ? "inf" : "-inf";
            if (value == 0)
                return 1 / value < 0 ? "-0.0" : "0.0";
            var r = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);
            string digits;
            int exponent;
            var mark = r.IndexOf('E');
            if (mark >= 0)
            {
                digits = r.Substring(0, mark).Replace(".", "");
                exponent = int.Parse(r.Substring(mark + 1), CultureInfo.InvariantCulture);
            }
            else
            {
                var dot = r.IndexOf('.');
                var whole = dot < 0 ? r : r.Substring(0, dot);
                var all = whole + (dot < 0 ? "" : r.Substring(dot + 1));
                var leading = all.Length - all.TrimStart('0').Length;
                digits = all.TrimStart('0');
                exponent = whole.Length - 1 - leading;
            }
            digits = digits.TrimEnd('0');
            if (digits.Length == 0)
                digits = "0";
            string body;
            if (exponent < -4 || exponent >= 16)
                body = digits[0] + (digits.Length > 1 ? "." + digits.Substring(1) : "") + "e" + (exponent < 0 ? "-" : "+") +
                       Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture);
            else if (exponent < 0)
                body = "0." + new string('0', -exponent - 1) + digits;
            else if (digits.Length <= exponent + 1)
                body = digits + new string('0', exponent + 1 - digits.Length) + ".0";
            else
                body = digits.Substring(0, exponent + 1) + "." + digits.Substring(exponent + 1);
            return (value < 0 ? "-" : "") + body;
        }

        private static string Trim(string number) => number.Contains('.') ? number.TrimEnd('0').TrimEnd('.') : number;
    }
}
