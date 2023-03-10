using System;


namespace Ann
{

    public static class Activations
    {

        public enum Activation { Sigmoid, ReLU, TanH }

        public static (Func<double, double>, Func<double, double>) GetActivationFunction(Activation activationFunction)
        {
            switch (activationFunction)
            {
                case Activation.Sigmoid:
                    return (Sigmoid, SigmoidDerivative);

                case Activation.ReLU:
                    return (ReLU, ReLUDerivative);

                case Activation.TanH:
                    return (TanH, TanHDerivative);
            }

            return (null, null);
        }

        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        public static double TanH(double x)
        {
            return 2 / (1 + Math.Exp(-2 * x)) - 1;
        }

        public static double TanHDerivative(double x)
        {
            return 1 - Math.Pow(TanH(x), 2);
        }
    }



}