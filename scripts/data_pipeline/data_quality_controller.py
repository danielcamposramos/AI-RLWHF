from typing import List, Dict, Tuple

class DataQualityController:
    """
    Ensures RLWHF tuple quality before training ingestion.

    This class provides a more advanced quality check than the simple gate,
    offering a quality score and detailed reports on the issues found in a dataset.
    """

    QUALITY_THRESHOLDS = {
        'min_reward_variance': 0.5,
        'max_duplicate_ratio': 0.1,
        'required_metadata_fields': ['source_ai', 'confidence_score', 'rubric_dimension']
    }

    def validate_tuples(self, tuples: List[Dict]) -> Tuple[bool, Dict[str, any]]:
        """
        Performs a comprehensive validation of RLWHF tuples and returns a detailed quality report.

        Args:
            tuples: A list of dictionaries, where each dictionary represents an RLWHF tuple.

        Returns:
            A tuple containing:
            - A boolean indicating if the dataset meets the quality threshold.
            - A dictionary containing a detailed quality report.
        """
        quality_report = {
            'total_tuples': len(tuples),
            'valid_tuples': 0,
            'quality_score': 0.0,
            'issues_found': [],
            'recommendations': []
        }

        if not tuples:
            return False, quality_report

        for tuple_data in tuples:
            validation_result = self._validate_single_tuple(tuple_data)
            if validation_result['is_valid']:
                quality_report['valid_tuples'] += 1
            else:
                quality_report['issues_found'].extend(validation_result['issues'])

        quality_report['quality_score'] = self._calculate_quality_score(quality_report)

        # Add recommendations based on the report
        if (quality_report['valid_tuples'] / quality_report['total_tuples']) < 0.8:
            quality_report['recommendations'].append("High number of invalid tuples. Review data formatting.")

        return quality_report['quality_score'] > 0.8, quality_report

    def _validate_single_tuple(self, tuple_data: Dict) -> Dict[str, any]:
        """Validates a single RLWHF tuple."""
        issues = []
        is_valid = True

        required_fields = ['prompt', 'answer', 'feedback', 'reward', 'metadata']
        for field in required_fields:
            if field not in tuple_data:
                issues.append(f"Missing required field: {field}")
                is_valid = False

        if 'metadata' in tuple_data:
            for field in self.QUALITY_THRESHOLDS['required_metadata_fields']:
                if field not in tuple_data['metadata']:
                    issues.append(f"Missing required metadata field: {field}")
                    is_valid = False

        if 'reward' in tuple_data and tuple_data['reward'] not in [-2, -1, 0, 1, 2]:
            issues.append(f"Invalid reward value: {tuple_data['reward']}")
            is_valid = False

        return {'is_valid': is_valid, 'issues': issues}

    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculates a quality score for the dataset based on the validation report."""
        if report['total_tuples'] == 0:
            return 0.0

        valid_ratio = report['valid_tuples'] / report['total_tuples']

        # Simple scoring logic: 1 point for valid ratio, can be expanded
        score = valid_ratio

        return score