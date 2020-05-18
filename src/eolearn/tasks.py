import itertools as it

import cv2
import numpy as np
from eolearn.core import EOTask, FeatureType
from eolearn.ml_tools.utilities import rolling_window
from scipy import ndimage


class CountValid(EOTask):
    """The task counts number of valid observations in time-series and stores
    the results in the timeless mask.
    """
    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch.add_feature(
            FeatureType.MASK_TIMELESS,
            self.name,
            np.count_nonzero(eopatch.mask[self.what], axis=0)
        )

        return eopatch


class TrainTestSplit(EOTask):
    """The task assigns the patch to 1 (train) or 0 (test) subset.
    """
    def __init__(self, feature_name):
        self.name = feature_name

    def execute(self, eopatch, subset):
        eopatch.add_feature(
            FeatureType.META_INFO,
            self.name,
            subset
        )

        return eopatch


class AddGradientTask(EOTask):
    """The task calculates inclination from DEM and ads it to eopatch.
    """
    def __init__(self, elevation_feature, result_feature):
        self.feature = elevation_feature
        self.result_feature = result_feature

    def execute(self, eopatch):
        elevation = eopatch[self.feature[0]][self.feature[1]].squeeze()
        gradient = ndimage.gaussian_gradient_magnitude(elevation, 1)
        eopatch.add_feature(
            self.result_feature[0],
            self.result_feature[1],
            gradient[..., np.newaxis]
        )

        return eopatch


###############################################################################
# Features
###############################################################################
# flake8: noqa


def normalize_feature(feature):
    """Normalize given feature.
    Assumes similar max and min throughout different features.

    :param feature: Feature to normalize
    :type feature: np.array
    :return: Normalized feature
    :rtype: np.array
    """
    f_min = np.min(feature)
    f_max = np.max(feature)
    if f_max != 0:
        return (feature - f_min) / (f_max - f_min)


def temporal_derivative(data, window_size=(3,)):
    padded_slope = np.zeros(data.shape)
    window = rolling_window(data, window_size, axes=0)

    slope = window[..., -1] - window[..., 0]  # TODO Missing division with time
    padded_slope[1:-1] = slope  # Padding with zeroes at the beginning and end

    return normalize_feature(padded_slope)


class AddBaseFeatures(EOTask):
    def __init__(self, bands_feature, band_names, features, c1=6, c2=7.5, L=1):
        self.bands_feature = bands_feature
        self.band_names = band_names
        self.features = features
        self.c1 = c1
        self.c2 = c2
        self.L = L

    def execute(self, eopatch):
        bands = self.band_names

        BLUE = eopatch.data[self.bands_feature][..., [bands.index('B02')]]
        GREEN = eopatch.data[self.bands_feature][..., [bands.index('B03')]]
        RED = eopatch.data[self.bands_feature][..., [bands.index('B04')]]
        NIR = eopatch.data[self.bands_feature][..., [bands.index('B08')]]

        if 'BLUE' in self.features:
            eopatch.add_feature(FeatureType.DATA, 'BLUE', BLUE)

        if 'GREEN' in self.features:
            eopatch.add_feature(FeatureType.DATA, 'GREEN', GREEN)
        
        if 'RED' in self.features:
            eopatch.add_feature(FeatureType.DATA, 'RED', RED)
        
        if 'NIR' in self.features:
            eopatch.add_feature(FeatureType.DATA, 'NIR', NIR)

        if 'NDVI' in self.features:
            NDVI = np.clip((NIR - RED) / (NIR + RED + 0.000000001), -1, 1)
            eopatch.add_feature(FeatureType.DATA, 'NDVI', NDVI)

        if 'NDVI_SLOPE' in self.features:
            # ASSUMES EVENLY SPACED
            NDVI_SLOPE = temporal_derivative(NDVI.squeeze())
            eopatch.add_feature(
                FeatureType.DATA,
                'NDVI_SLOPE',
                NDVI_SLOPE[..., np.newaxis]
            )

        if 'NDWI' in self.features:
            band_a = eopatch.data[self.bands_feature][..., 1]
            band_b = eopatch.data[self.bands_feature][..., 3]
            NDWI = np.clip((band_a - band_b) / (band_a + band_b + 0.000000001), -1, 1)
            eopatch.add_feature(
                FeatureType.DATA,
                'NDWI',
                NDWI[..., np.newaxis]
            )

        if 'EVI' in self.features:
            EVI = np.clip(2.5 * ((NIR - RED) / (NIR + (self.c1 * RED) - (self.c2 * BLUE) + self.L + 0.000000001)), -1, 1)
            eopatch.add_feature(FeatureType.DATA, 'EVI', EVI)

        if 'EVI_SLOPE' in self.features:
            EVI_SLOPE = temporal_derivative(EVI.squeeze())
            eopatch.add_feature(
                FeatureType.DATA,
                'EVI_SLOPE',
                EVI_SLOPE[..., np.newaxis]
            )

        if 'SAVI' in self.features:
            l_var = 0.5
            SAVI = np.clip(((NIR - RED) / (NIR + RED + l_var + 0.000000001)) * (1 + l_var), -1, 1)
            eopatch.add_feature(FeatureType.DATA, 'SAVI', SAVI)

        if 'SIPI' in self.features:
            # TODO: Better division by 0 handling
            SIPI = np.clip((NIR - BLUE) / (NIR - RED + 0.000000001), 0, 2)
            eopatch.add_feature(FeatureType.DATA, 'SIPI', SIPI)

        if 'ARVI' in self.features:
            # TODO: Better division by 0 handling
            ARVI = np.clip((NIR - (2 * RED) + BLUE) / (NIR + (2 * RED) + BLUE + 0.000000001), -1, 1)
            eopatch.add_feature(FeatureType.DATA, 'ARVI', ARVI)

        if 'ARVI_SLOPE' in self.features:
            ARVI_SLOPE = temporal_derivative(ARVI.squeeze())
            eopatch.add_feature(
                FeatureType.DATA,
                'ARVI_SLOPE',
                ARVI_SLOPE[..., np.newaxis]
            )

        return eopatch


class AddStreamTemporalFeaturesTask(EOTask):
    # pylint: disable=too-many-instance-attributes
    """ Task that implements and adds to eopatch the spatio-temporal features proposed in [1].
    The features are added to the `data_timeless` attribute dictionary of eopatch.
    [1] Valero et. al. "Production of adynamic cropland mask by processing remote sensing
    image series at high temporal and spatial resolutions" Remote Sensing, 2016.
    """

    def __init__(self, data_feature=(FeatureType.DATA, 'NDVI'), data_index=None,
                 ndvi_feature_name=(FeatureType.DATA, 'NDVI'), mask_data=True, *,
                 max_val_feature='max_val', min_val_feature='min_val', mean_val_feature='mean_val',
                 sd_val_feature='sd_val', diff_max_feature='diff_max', diff_min_feature='diff_min',
                 diff_diff_feature='diff_diff', max_mean_feature='max_mean_feature',
                 max_mean_len_feature='max_mean_len', max_mean_surf_feature='max_mean_surf',
                 pos_surf_feature='pos_surf', pos_len_feature='pos_len', pos_rate_feature='pos_rate',
                 neg_surf_feature='neg_surf', neg_len_feature='neg_len', neg_rate_feature='neg_rate',
                 pos_transition_feature='pos_tran', neg_transition_feature='neg_tran',
                 feature_name_prefix=None, window_size=2, interval_tolerance=0.1, base_surface_min=-1.,
                 ndvi_barren_soil_cutoff=0.1):
        """
        :param data_feature: Name of data feature with values that are considered. Default is `'NDVI'`
        :type data_feature: object
        :param data_index: Index of to be extracted from last dimension in `data_feature`. If None, last dimension of
            data array is assumed ot be of size 1 (e.g. as in NDVI). Default is `None`
        :type data_index: int
        :param ndvi_feature_name: Name of data feature with NDVI values for bare soil transition considerations.
        If None, soil transitions are not calculated and set as 0
        :type ndvi_feature_name: obj
        :param mask_data: Flag specifying whether to mask data with `'VALID_DATA'` mask. If `False`, the `'IS_DATA'`
                          mask is used
        :param max_val_feature: Name of feature with computed max
        :type max_val_feature: str
        :param min_val_feature: Name of feature with computed min
        :type min_val_feature: str
        :param mean_val_feature: Name of feature with computed mean
        :type mean_val_feature: str
        :param sd_val_feature: Name of feature with computed standard deviation
        :param sd_val_feature: str
        :param diff_max_feature: Name of feature with computed max difference in a temporal sliding window
        :param diff_max_feature: str
        :param diff_min_feature: Name of feature with computed min difference in a temporal sliding window
        :param diff_min_feature: str
        :param diff_diff_feature: Name of feature with computed difference of difference in a temporal sliding window
        :param diff_diff_feature: str
        :param max_mean_feature: Name of feature with computed max of mean in a sliding window
        :param max_mean_feature: str
        :param max_mean_len_feature: Name of feature with computed length of time interval corresponding to max_mean
        :param max_mean_len_feature: str
        :param max_mean_surf_feature: Name of feature with computed surface under curve corresponding to max_mean
        :param max_mean_surf_feature: str
        :param pos_surf_feature: Name of feature with computed largest surface under curve where first derivative
        is positive
        :param pos_surf_feature: str
        :param pos_len_feature: Name of feature with computed length of time interval corresponding to pos_surf
        :param pos_len_feature: str
        :param pos_rate_feature: Name of feature with computed rate of change corresponding to pos_surf
        :param pos_rate_feature: str
        :param neg_surf_feature: Name of feature with computed largest surface under curve where first derivative
        is negative
        :param neg_surf_feature: str
        :param neg_len_feature: Name of feature with computed length of time interval corresponding to neg_surf
        :param neg_len_feature: str
        :param neg_rate_feature: Name of feature with computed rate of change corresponding to neg_surf
        :param neg_rate_feature: str
        :param pos_transition_feature: Name of feature to be associated to computed feature of argmax values
        :param pos_transition_feature: str
        :param neg_transition_feature: Name of feature to be associated to computed feature of argmax values
        :param neg_transition_feature: str
        :param feature_name_prefix: String to be used as prefix in names for calculated features.
        Default: value of data_feature
        :param feature_name_prefix: str
        :param window_size: Size of sliding temporal window
        :param window_size: int
        :param interval_tolerance: Tolerance for calculation of max_mean family of data features
        :param interval_tolerance: float
        :param base_surface_min: Minimal base value for data, used to more accurately calculate surface under curve.
        Default for indices like values is -1.0.
        :param base_surface_min: float
        :param ndvi_barren_soil_cutoff: Cutoff for bare soil detection
        :type ndvi_barren_soil_cutoff: 0.1
        """
        # pylint: disable=too-many-locals
        self.data_feature = next(iter(self._parse_features(data_feature, default_feature_type=FeatureType.DATA)))
        self.data_index = data_index or 0
        self.mask_data = mask_data
        self.ndvi_feature_name = next(iter(self._parse_features(ndvi_feature_name,
                                                                default_feature_type=FeatureType.DATA)))

        if feature_name_prefix:
            self.feature_name_prefix = feature_name_prefix
            if not feature_name_prefix.endswith("_"):
                self.feature_name_prefix += "_"
        else:
            self.feature_name_prefix = data_feature + "_"

        self.max_val_feature = self.feature_name_prefix + max_val_feature
        self.min_val_feature = self.feature_name_prefix + min_val_feature
        self.mean_val_feature = self.feature_name_prefix + mean_val_feature
        self.sd_val_feature = self.feature_name_prefix + sd_val_feature
        self.diff_max_feature = self.feature_name_prefix + diff_max_feature
        self.diff_min_feature = self.feature_name_prefix + diff_min_feature
        self.diff_diff_feature = self.feature_name_prefix + diff_diff_feature
        self.max_mean_feature = self.feature_name_prefix + max_mean_feature
        self.max_mean_len_feature = self.feature_name_prefix + max_mean_len_feature
        self.max_mean_surf_feature = self.feature_name_prefix + max_mean_surf_feature
        self.pos_surf_feature = self.feature_name_prefix + pos_surf_feature
        self.pos_len_feature = self.feature_name_prefix + pos_len_feature
        self.pos_rate_feature = self.feature_name_prefix + pos_rate_feature
        self.neg_surf_feature = self.feature_name_prefix + neg_surf_feature
        self.neg_len_feature = self.feature_name_prefix + neg_len_feature
        self.neg_rate_feature = self.feature_name_prefix + neg_rate_feature
        self.pos_transition_feature = self.feature_name_prefix + pos_transition_feature
        self.neg_transition_feature = self.feature_name_prefix + neg_transition_feature

        self.window_size = window_size
        self.interval_tolerance = interval_tolerance
        self.base_surface_min = base_surface_min

        self.ndvi_barren_soil_cutoff = ndvi_barren_soil_cutoff

    def execute(self, eopatch):
        """ Compute spatio-temporal features for input eopatch
        :param eopatch: Input eopatch
        :return: eopatch with computed spatio-temporal features
        """
        # pylint: disable=invalid-name
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        all_dates = np.asarray([x.toordinal() for x in eopatch.timestamp])

        data = eopatch[self.data_feature[0]][self.data_feature[1]][..., self.data_index]
        valid_data_mask = np.ones_like(data)

        if data.ndim == 3:
            _, h, w = data.shape
        else:
            raise ValueError('{} feature has incorrect number of dimensions'.format(self.data_feature))

        madata = np.ma.array(data, dtype=np.float32, mask=~valid_data_mask.astype(np.bool))

        # Vectorized
        data_max_val = np.ma.MaskedArray.max(madata, axis=0).filled()
        data_min_val = np.ma.MaskedArray.min(madata, axis=0).filled()
        data_mean_val = np.ma.MaskedArray.mean(madata, axis=0).filled()
        data_sd_val = np.ma.MaskedArray.std(madata, axis=0).filled()

        data_diff_max = np.empty((h, w))
        data_diff_min = np.empty((h, w))
        # data_diff_diff = np.empty((h, w)) # Calculated later

        data_max_mean = np.empty((h, w))
        data_max_mean_len = np.empty((h, w))
        data_max_mean_surf = np.empty((h, w))

        data_pos_surf = np.empty((h, w))
        data_pos_len = np.empty((h, w))
        data_pos_rate = np.empty((h, w))

        data_neg_surf = np.empty((h, w))
        data_neg_len = np.empty((h, w))
        data_neg_rate = np.empty((h, w))

        data_pos_tr = np.empty((h, w))
        data_neg_tr = np.empty((h, w))
        for ih, iw in it.product(range(h), range(w)):
            data_curve = madata[:, ih, iw]
            valid_idx = np.where(~madata.mask[:, ih, iw])[0]

            data_curve = data_curve[valid_idx].filled()

            valid_dates = all_dates[valid_idx]

            sw_max = np.max(rolling_window(data_curve, self.window_size), -1)
            sw_min = np.min(rolling_window(data_curve, self.window_size), -1)

            sw_diff = sw_max - sw_min

            data_diff_max[ih, iw] = np.max(sw_diff)
            data_diff_min[ih, iw] = np.min(sw_diff)

            sw_mean = np.mean(rolling_window(data_curve, self.window_size), -1)
            max_mean = np.max(sw_mean)

            data_max_mean[ih, iw] = max_mean

            # Calculate max mean interval
            # Work with mean windowed or whole set?
            workset = data_curve  # or sw_mean, which is a bit more smoothed
            higher_mask = workset >= max_mean - ((1-self.interval_tolerance) * abs(max_mean))

            # Just normalize to have 0 on each side
            higher_mask_norm = np.zeros(len(higher_mask) + 2)
            higher_mask_norm[1:len(higher_mask)+1] = higher_mask

            # index of 1 that have 0 before them, SHIFTED BY ONE TO RIGHT
            up_mask = (higher_mask_norm[1:] == 1) & (higher_mask_norm[:-1] == 0)

            # Index of 1 that have 0 after them, correct indices
            down_mask = (higher_mask_norm[:-1] == 1) & (higher_mask_norm[1:] == 0)

            # Calculate length of interval as difference between times of first and last high enough observation,
            # in particular, if only one such observation is high enough, the length of such interval is 0
            # One can extend this to many more ways of calculating such length:
            # take forward/backward time differences, interpolate in between (again...) and treat this as
            # continuous problem, take mean of the time intervals between borders...
            times_up = valid_dates[up_mask[:-1]]
            times_down = valid_dates[down_mask[1:]]

            # There may be several such intervals, take the longest one
            times_diff = times_down - times_up
            # if there are no such intervals, the signal is constant,
            # set everything to zero and continue
            if times_diff.size == 0:
                data_max_mean_len[ih, iw] = 0
                data_max_mean_surf[ih, iw] = 0

                data_pos_surf[ih, iw] = 0
                data_pos_len[ih, iw] = 0
                data_pos_rate[ih, iw] = 0

                data_neg_surf[ih, iw] = 0
                data_neg_len[ih, iw] = 0
                data_neg_rate[ih, iw] = 0

                if self.ndvi_feature_name:
                    data_pos_tr[ih, iw] = 0
                    data_neg_tr[ih, iw] = 0
                continue

            max_ind = np.argmax(times_diff)
            data_max_mean_len[ih, iw] = times_diff[max_ind]

            fst = np.where(up_mask[:-1])[0]
            snd = np.where(down_mask[1:])[0]

            surface = np.trapz(data_curve[fst[max_ind]:snd[max_ind]+1] - self.base_surface_min,
                               valid_dates[fst[max_ind]:snd[max_ind]+1])
            data_max_mean_surf[ih, iw] = surface

            # Derivative based features
            # How to approximate derivative?
            derivatives = np.gradient(data_curve, valid_dates)

            # Positive derivative
            pos = np.zeros(len(derivatives) + 2)
            pos[1:len(derivatives)+1] = derivatives >= 0

            pos_der_int, pos_der_len, pos_der_rate, (start, _) = \
                self.derivative_features(pos, valid_dates, data_curve, self.base_surface_min)

            data_pos_surf[ih, iw] = pos_der_int
            data_pos_len[ih, iw] = pos_der_len
            data_pos_rate[ih, iw] = pos_der_rate

            neg = np.zeros(len(derivatives) + 2)
            neg[1:len(derivatives)+1] = derivatives <= 0

            neg_der_int, neg_der_len, neg_der_rate, (_, end) = \
                self.derivative_features(neg, valid_dates, data_curve, self.base_surface_min)

            data_neg_surf[ih, iw] = neg_der_int
            data_neg_len[ih, iw] = neg_der_len
            data_neg_rate[ih, iw] = neg_der_rate

            if self.ndvi_feature_name:
                data_pos_tr[ih, iw] = \
                    np.any(eopatch[self.ndvi_feature_name[0]][self.ndvi_feature_name[1]][:start+1, ih, iw, 0] <=
                           self.ndvi_barren_soil_cutoff)
                data_neg_tr[ih, iw] = \
                    np.any(eopatch[self.ndvi_feature_name[0]][self.ndvi_feature_name[1]][end:, ih, iw, 0] <=
                           self.ndvi_barren_soil_cutoff)

        eopatch.data_timeless[self.max_val_feature] = data_max_val[..., np.newaxis]
        eopatch.data_timeless[self.min_val_feature] = data_min_val[..., np.newaxis]
        eopatch.data_timeless[self.mean_val_feature] = data_mean_val[..., np.newaxis]
        eopatch.data_timeless[self.sd_val_feature] = data_sd_val[..., np.newaxis]

        eopatch.data_timeless[self.diff_max_feature] = data_diff_max[..., np.newaxis]
        eopatch.data_timeless[self.diff_min_feature] = data_diff_min[..., np.newaxis]
        eopatch.data_timeless[self.diff_diff_feature] = (data_diff_max - data_diff_min)[..., np.newaxis]

        eopatch.data_timeless[self.max_mean_feature] = data_max_mean[..., np.newaxis]
        eopatch.data_timeless[self.max_mean_len_feature] = data_max_mean_len[..., np.newaxis]
        eopatch.data_timeless[self.max_mean_surf_feature] = data_max_mean_surf[..., np.newaxis]

        eopatch.data_timeless[self.pos_len_feature] = data_pos_len[..., np.newaxis]
        eopatch.data_timeless[self.pos_surf_feature] = data_pos_surf[..., np.newaxis]
        eopatch.data_timeless[self.pos_rate_feature] = data_pos_rate[..., np.newaxis]
        eopatch.data_timeless[self.pos_transition_feature] = data_pos_tr[..., np.newaxis]

        eopatch.data_timeless[self.neg_len_feature] = data_neg_len[..., np.newaxis]
        eopatch.data_timeless[self.neg_surf_feature] = data_neg_surf[..., np.newaxis]
        eopatch.data_timeless[self.neg_rate_feature] = data_neg_rate[..., np.newaxis]
        eopatch.data_timeless[self.neg_transition_feature] = data_neg_tr[..., np.newaxis]

        return eopatch

    def get_data(self, patch):
        """Extracts and concatenates newly extracted features contained in the provided eopatch
        :param patch: Input eopatch
        :type patch: eolearn.core.EOPatch
        :return: Tuple of two lists: names of extracted features and their values
        """
        names = [self.max_val_feature, self.min_val_feature, self.mean_val_feature, self.sd_val_feature,
                 self.diff_max_feature, self.diff_min_feature, self.diff_diff_feature,
                 self.max_mean_feature, self.max_mean_len_feature, self.max_mean_surf_feature,
                 self.pos_len_feature, self.pos_surf_feature, self.pos_rate_feature, self.pos_transition_feature,
                 self.neg_len_feature, self.neg_surf_feature, self.neg_rate_feature, self.neg_transition_feature]

        dim_x, dim_y, _ = patch.data_timeless[names[0]].shape

        data = np.zeros((dim_x, dim_y, len(names)))
        for ind, name in enumerate(names):
            data[..., ind] = patch.data_timeless[name].squeeze()

        return names, data

    @staticmethod
    def derivative_features(mask, valid_dates, data, base_surface_min):
        """Calculates derivative based features for provided data points selected by
        mask (increasing data points, decreasing data points)
        :param mask: Mask indicating data points considered
        :type mask: np.array
        :param valid_dates: Dates (x-axis for surface calculation)
        :type valid_dates: np.array
        :param data: Base data
        :type data: np.array
        :param base_surface_min: Base surface value (added to each measurement)
        :type base_surface_min: float
        :return: Tuple of: maximal consecutive surface under the data curve,
                           date length corresponding to maximal surface interval,
                           rate of change in maximal interval,
                           (starting date index of maximal interval, ending date index of interval)
        """
        # index of 1 that have 0 before them, shifted by one to right
        up_mask = (mask[1:] == 1) & (mask[:-1] == 0)

        # Index of 1 that have 0 after them, correct indices
        down_mask = (mask[:-1] == 1) & (mask[1:] == 0)

        fst_der = np.where(up_mask[:-1])[0]
        snd_der = np.where(down_mask[1:])[0]
        der_ind_max = -1
        der_int_max = -1

        for ind, (start, end) in enumerate(zip(fst_der, snd_der)):

            integral = np.trapz(
                data[start:end + 1] - base_surface_min,
                valid_dates[start:end + 1])

            if abs(integral) >= abs(der_int_max):
                der_int_max = integral
                der_ind_max = ind

        start_ind = fst_der[der_ind_max]
        end_ind = snd_der[der_ind_max]

        der_len = valid_dates[end_ind] - valid_dates[start_ind]
        der_rate = (data[end_ind] - data[start_ind]) / der_len if der_len else 0

        return der_int_max, der_len, der_rate, (start_ind, end_ind)


###############################################################################
# Edge extracrtion
###############################################################################


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class AddGray(EOTask):
    def __init__(self, bands_feature):
        self.bands_feature = bands_feature

    def execute(self, eopatch):
        img = np.clip(eopatch.data[self.bands_feature][..., [2, 1, 0]] * 3.5, 0, 1)
        t, w, h, _ = img.shape
        gray_img = np.zeros((t, w, h))
        for time in range(t):
            img0 = np.clip(eopatch[FeatureType.DATA][self.bands_feature][time][..., [2, 1, 0]] * 3.5, 0, 1)
            img = rgb2gray(img0)
            gray_img[time] = (img * 255).astype(np.uint8)

        eopatch.add_feature(FeatureType.DATA, 'GRAY', gray_img[..., np.newaxis])
        return eopatch


class ExtractEdgesTask(EOTask):

    def __init__(self,
                 edge_features,
                 structuring_element,
                 excluded_features,
                 dilation_mask,
                 erosion_mask,
                 output_feature,
                 adjust_function,
                 adjust_threshold,
                 yearly_low_threshold):

        self.edge_features = edge_features
        self.structuring_element = structuring_element
        self.excluded_features = excluded_features
        self.dilation_mask = dilation_mask
        self.erosion_mask = erosion_mask
        self.output_feature = output_feature
        self.adjust_function = adjust_function
        self.adjust_threshold = adjust_threshold
        self.yearly_low_threshold = yearly_low_threshold

    def extract_edges(self, eopatch, feature_type, feature_name, low_threshold, high_threshold, blur):

        image = eopatch[feature_type][feature_name]
        t, w, h, _ = image.shape
        all_edges = np.zeros((t, w, h))
        for time in range(t):
            image_one = image[time]
            edge = self.one_edge(image_one, low_threshold, high_threshold, blur)
            all_edges[time] = edge
        #eopatch.add_feature(FeatureType.MASK, feature_name + '_EDGE', all_edges[..., np.newaxis])
        return all_edges

    def one_edge(self, image, low_threshold, high_threshold, blur):
        ##########QUICK NORMALIZATION -  SHOULD BE LATER IMPROVED / MOVED SOMEWHERE ELSE
        f_min = np.min(image)
        f_max = np.max(image)
        image = (image - f_min) / f_max * 255
        image = image.squeeze()
        kernel_size, sigma = blur
        smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma)
        edges = cv2.Canny(smoothed_image.astype(np.uint8), low_threshold, high_threshold)
        return edges > 0

    def filter_unwanted_areas(self, eopatch, feature, threshold):
        # Returns mask of areas that should be excluded (low NDVI etc...)
        bands = eopatch[feature[0]][feature[1]]
        t, w, h, _ = bands.shape
        mask = np.zeros((w, h))

        for time in range(t):
            fet = eopatch[feature[0]][feature[1]][time].squeeze()
            mask_cur = fet <= threshold
            mask_cur = cv2.dilate((mask_cur * 255).astype(np.uint8), self.dilation_mask * 255)
            mask_cur = cv2.erode((mask_cur * 255).astype(np.uint8), self.erosion_mask * 255)
            mask = mask + mask_cur

        mask = (mask / t) > self.yearly_low_threshold
        #eopatch.add_feature(FeatureType.MASK_TIMELESS, 'LOW_' + feature[1], mask[..., np.newaxis])
        return mask

    def normalize_feature(self, feature):
        f_min = np.min(feature)
        f_max = np.max(feature)
        return (feature - f_min) / (f_max - f_min)

    def execute(self, eopatch):

        bands = eopatch.data['BANDS']
        t, w, h, _ = bands.shape

        no_feat = len(self.edge_features)
        edge_vector = np.zeros((no_feat, t, w, h))
        for i in range(no_feat):
            arg = self.edge_features[i]
            one_edge = self.extract_edges(eopatch, arg['FeatureType'], arg['FeatureName'],
                                          arg['CannyThresholds'][0], arg['CannyThresholds'][1], arg['BlurArguments'])
            v1 = eopatch[arg['FeatureType']][arg['FeatureName']].squeeze()
            v1 = self.normalize_feature(v1)
            v1 = [self.adjust_function(x) for x in v1]
            edge_vector[i] = one_edge * v1

        edge_vector1 = np.sum(edge_vector, (0, 1))
        edge_vector1 = edge_vector1 / (t * len(self.edge_features))
        edge_vector = edge_vector1 > self.adjust_threshold

        for unwanted, threshold in self.excluded_features:
            mask = self.filter_unwanted_areas(eopatch, unwanted, threshold)

            edge_vector = np.logical_or(edge_vector, mask)

        edge_vector = 1 - edge_vector
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.output_feature[1], edge_vector[..., np.newaxis])
        return eopatch
