import unittest
import numpy as np
from run_analysis import RetailData,allocate_proportional,optimise_orders,forecast_metrics

class ForecastChecks(unittest.TestCase):
    def test_future_sales_cannot_change_origin_features(self):
        data=RetailData();before=data.frame(1829)
        data.y[:,1829:]=999999
        after=data.frame(1829)
        np.testing.assert_allclose(before.to_numpy(),after.to_numpy())

    def test_seasonal_naive_uses_observed_week_only(self):
        data=RetailData();expected=np.tile(data.y[:,1822:1829],(1,4))
        np.testing.assert_array_equal(data.forecast(1829,'Weekly seasonal naive'),expected)

    def test_perfect_forecast_scores_zero_error(self):
        actual=np.array([[1.,2.,3.],[3.,1.,2.]])
        metrics,_=forecast_metrics(actual,actual,actual)
        self.assertEqual(metrics['wape'],0);self.assertEqual(metrics['mean_rmsse'],0)

    def test_proportional_orders_respect_cash(self):
        q=allocate_proportional(np.array([10.,20.,5.]),np.array([2.,3.,4.]),25.)
        self.assertTrue((q>=0).all());self.assertLessEqual(q@np.array([2.,3.,4.]),25.)
        np.testing.assert_array_equal(allocate_proportional(np.ones(3),np.ones(3),0),0)

    def test_optimizer_known_small_problem(self):
        scenarios=np.tile(np.array([2.,3.]),(10,1));cost=np.array([1.,1.])
        q,status=optimise_orders(np.zeros(2),scenarios,cost,np.array([2.,2.]),5.,7)
        np.testing.assert_array_equal(q,[2,3]);self.assertEqual(status['status'],0)
        q,_=optimise_orders(np.zeros(2),scenarios,cost,np.array([2.,2.]),0.,7)
        np.testing.assert_array_equal(q,0)

if __name__=='__main__':unittest.main()
