from pathlib import Path

import numpy as np
import pandas as pd
import seaborn

from .util import load_gait_phase_data, load_gait_phase_data_old


def generate_gait_data_report_old(data_path, save_path, report_title, feature_l=None):
    """Generate a basic data analysis report."""
    from pandas_profiling import ProfileReport

    if feature_l is None:
        feature_l = [
            "right_hip",
            "left_hip",
            "right_knee",
            "left_knee",
            "cop_right_x",
            "cop_left_x",
            "cop_right_y",
            "cop_left_y",
            "cop_right_sum",
            "cop_left_sum",
        ]
    _, data = load_gait_phase_data_old(data_path, keys=feature_l)
    df = pd.DataFrame(data=data, columns=feature_l)
    profile = df.profile_report(title=report_title, progress_bar=True)
    profile.to_file(output_file=save_path)


def generate_gait_data_report(data_path, save_path, report_title, feature_l=None):
    """Generate a basic data analysis report."""
    from pandas_profiling import ProfileReport

    if feature_l is None:
        feature_l = [
            "right_hip",
            "left_hip",
            "right_knee",
            "left_knee",
            "cop_right_x",
            "cop_left_x",
            "cop_right_y",
            "cop_left_y",
            "cop_right_sum",
            "cop_left_sum",
        ]
    _, data = load_gait_phase_data(data_path, keys=feature_l)
    df = pd.DataFrame(data=data, columns=feature_l)
    profile = df.profile_report(title=report_title, progress_bar=True)
    profile.to_file(output_file=save_path)

