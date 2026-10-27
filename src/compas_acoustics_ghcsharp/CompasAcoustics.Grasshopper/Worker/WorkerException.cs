using System;
using System.Linq;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>An error of the worker, or of starting it: the message is what a component shows.</summary>
    public sealed class WorkerException : Exception
    {
        /// <summary>The HTTP status the worker answered with; null when the worker did not answer.</summary>
        public int? Status { get; }

        /// <summary>The worker's whole text, e.g. a Python traceback whose last line is the message.</summary>
        public string? Details { get; }

        public WorkerException(string message, int? status = null, string? details = null) : base(message)
        {
            Status = status;
            Details = details;
        }

        /// <summary>The line of a Python traceback that names the error, or the text itself.</summary>
        public static string Summary(string text)
        {
            if (!text.Contains("Traceback (most recent call last)"))
                return text;
            return text.Split('\n').Select(line => line.TrimEnd()).LastOrDefault(line => line.Length > 0) ?? text;
        }

        public static WorkerException FromAnswer(int status, string text) => new WorkerException(Summary(text), status, text);
    }
}
