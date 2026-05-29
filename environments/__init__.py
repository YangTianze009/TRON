"""TRON: Targeted Rule-Verifiable Online Environments for Visual Reasoning RL.

This package contains the 520 procedural visual reasoning environments
described in the TRON paper. Each environment is a generator-verifier
pair that produces fresh image-question-answer rollouts with deterministic
rewards.

Environments are organized into five ability buckets (see ../buckets/):
  - spatial      (111 envs): 3D rotation, cube nets, navigation, perspective
  - math         (131 envs): geometry, analytic geometry, algebra, probability
  - diagram      (144 envs): charts, tables, graphs, flowcharts, scientific figs
  - pattern      (104 envs): constraint puzzles, analogies, sequences, planning
  - count        ( 30 envs): visual enumeration, path counting, measurement

Usage:
    from TRON.environments import ENVIRONMENTS
    cls = ENVIRONMENTS["aquarium_grid"]
    env = cls()
    sample = env.generate(level=3)   # returns (image, question, answer)
"""

ENVIRONMENTS = {}
BUCKET = {}

def _register(name, cls, bucket=None):
    ENVIRONMENTS[name] = cls
    if bucket is not None:
        BUCKET[name] = bucket


from .algebra_robustness_qa import AlgebraRobustnessQA
_register("algebra_robustness", AlgebraRobustnessQA, "math")
from .ambiguous_label_resolution_qa import AmbiguousLabelResolutionQA
_register("ambiguous_label_resolution", AmbiguousLabelResolutionQA, "pattern")
from .analogy_from_sequence_qa import AnalogyFromSequenceQA
_register("analogy_from_sequence", AnalogyFromSequenceQA, "pattern")
from .analogy_multiple_dimensions_qa import AnalogyMultipleDimensionsQA
_register("analogy_multiple_dimensions", AnalogyMultipleDimensionsQA, "pattern")
from .analogy_with_negation_qa import AnalogyWithNegationQA
_register("analogy_with_negation", AnalogyWithNegationQA, "pattern")
from .analytic_geom_chain_qa import AnalyticGeomChainQA
_register("analytic_geom_chain", AnalyticGeomChainQA, "math")
from .analytic_geometry_visual_qa import AnalyticGeometryVisualQA
_register("analytic_geometry_visual", AnalyticGeometryVisualQA, "math")
from .angle_bisector_chain_deep_qa import AngleBisectorChainDeepQA
_register("angle_bisector_chain_deep", AngleBisectorChainDeepQA, "math")
from .angle_bisector_chain_qa import AngleBisectorChainQA
_register("angle_bisector_chain", AngleBisectorChainQA, "math")
from .angle_bisector_qa import AngleBisectorQA
_register("angle_bisector", AngleBisectorQA, "math")
from .angle_chain_qa import AngleChainQA
_register("angle_chain", AngleChainQA, "math")
from .angle_chase_minimal_text_qa import AngleChaseMinimalTextQA
_register("angle_chase_minimal_text", AngleChaseMinimalTextQA, "math")
from .angle_chase_two_step_qa import AngleChaseTwoStepQA
_register("angle_chase_two_step", AngleChaseTwoStepQA, "math")
from .aquarium_grid_qa import AquariumGridQA
_register("aquarium_grid", AquariumGridQA, "spatial")
from .arc_angle_relationship_qa import ArcAngleRelationshipQA
_register("arc_angle_relationship", ArcAngleRelationshipQA, "math")
from .area_chart_qa import AreaChartQA
_register("area_chart", AreaChartQA, "diagram")
from .area_decomposition_qa import AreaDecompositionQA
_register("area_decomposition", AreaDecompositionQA, "math")
from .area_under_curve_estimation_qa import AreaUnderCurveEstimationQA
_register("area_under_curve_estimation", AreaUnderCurveEstimationQA, "count")
from .argument_contradiction_qa import ArgumentContradictionQA
_register("argument_contradiction", ArgumentContradictionQA, "pattern")
from .arrow_block_simulate_qa import ArrowBlockSimulateQA
_register("arrow_block_simulate", ArrowBlockSimulateQA, "spatial")
from .arrow_count_flow_qa import ArrowCountFlowQA
_register("arrow_count_flow", ArrowCountFlowQA, "count")
from .arrow_grid_missing_qa import ArrowGridMissingQA
_register("arrow_grid_missing", ArrowGridMissingQA, "spatial")
from .asymmetric_shape_rotation_qa import AsymmetricShapeRotationQA
_register("asymmetric_shape_rotation", AsymmetricShapeRotationQA, "spatial")
from .attribute_analogy_verbal_qa import AttributeAnalogyVerbalQA
_register("attribute_analogy_verbal", AttributeAnalogyVerbalQA, "pattern")
from .attribute_count_quantize_qa import AttributeCountQuantizeQA
_register("attribute_count_quantize", AttributeCountQuantizeQA, "count")
from .attribute_enumeration_discovery_qa import AttributeEnumerationDiscoveryQA
_register("attribute_enumeration_discovery", AttributeEnumerationDiscoveryQA, "math")
from .attribute_grouping_metalist_qa import AttributeGroupingMetalistQA
_register("attribute_grouping_metalist", AttributeGroupingMetalistQA, "pattern")
from .attribute_height_quantize_qa import AttributeHeightQuantizeQA
_register("attribute_height_quantize", AttributeHeightQuantizeQA, "math")
from .attribute_ordering_qa import AttributeOrderingQA
_register("attribute_ordering", AttributeOrderingQA, "math")
from .bar_chart_aggregate_qa import BarChartAggregateQA
_register("bar_chart_aggregate", BarChartAggregateQA, "diagram")
from .bar_chart_compare_qa import BarChartCompareQA
_register("bar_chart_compare", BarChartCompareQA, "diagram")
from .bar_chart_position_claim_qa import BarChartPositionClaimQA
_register("bar_chart_position_claim", BarChartPositionClaimQA, "diagram")
from .bearing_compass_qa import BearingCompassQA
_register("bearing_compass", BearingCompassQA, "spatial")
from .before_after_qa import BeforeAfterQA
_register("before_after", BeforeAfterQA, "pattern")
from .bennett_mechanical_qa import BennettMechanicalQA
_register("bennett_mechanical", BennettMechanicalQA, "math")
from .binairo_solve_qa import BinairoSolveQA
_register("binairo_solve", BinairoSolveQA, "pattern")
from .binary_search_tree_operation_qa import BinarySearchTreeOperationQA
_register("binary_search_tree_operation", BinarySearchTreeOperationQA, "diagram")
from .bipartite_judge_qa import BipartiteJudgeQA
_register("bipartite_judge", BipartiteJudgeQA, "diagram")
from .blank_annotation_trap_qa import BlankAnnotationTrapQA
_register("blank_annotation_trap", BlankAnnotationTrapQA, "pattern")
from .block_assembly_qa import BlockAssemblyQA
_register("block_assembly", BlockAssemblyQA, "spatial")
from .boarding_pass_duration_qa import BoardingPassDurationQA
_register("boarding_pass_duration", BoardingPassDurationQA, "math")
from .bottle_water_level_qa import BottleWaterLevelQA
_register("bottle_water_level", BottleWaterLevelQA, "math")
from .box_plot_comparison_qa import BoxPlotComparisonQA
_register("box_plot_comparison", BoxPlotComparisonQA, "diagram")
from .bubble_chart_qa import BubbleChartQA
_register("bubble_chart", BubbleChartQA, "diagram")
from .business_table_qa import BusinessTableQA
_register("business_table", BusinessTableQA, "diagram")
from .calcudoku_qa import CalcudokuQA
_register("calcudoku", CalcudokuQA, "pattern")
from .calendar_date_qa import CalendarDateQA
_register("calendar_date", CalendarDateQA, "math")
from .camera_rotation_direction_qa import CameraRotationDirectionQA
_register("camera_rotation_direction", CameraRotationDirectionQA, "spatial")
from .campsite_solve_qa import CampsiteSolveQA
_register("campsite_solve", CampsiteSolveQA, "pattern")
from .categorical_syllogism_adversarial_qa import CategoricalSyllogismAdversarialQA
_register("categorical_syllogism_adversarial", CategoricalSyllogismAdversarialQA, "pattern")
from .causal_diagram_qa import CausalDiagramQA
_register("causal_diagram", CausalDiagramQA, "diagram")
from .chart_aggregate_claim_qa import ChartAggregateClaimQA
_register("chart_aggregate_claim", ChartAggregateClaimQA, "diagram")
from .chart_claim_verify_qa import ChartClaimVerifyQA
_register("chart_claim_verify", ChartClaimVerifyQA, "diagram")
from .chart_closest_match_qa import ChartClosestMatchQA
_register("chart_closest_match", ChartClosestMatchQA, "diagram")
from .chart_comparison_qa import ChartComparisonQA
_register("chart_comparison", ChartComparisonQA, "diagram")
from .chart_conversational_argmax_qa import ChartConversationalArgmaxQA
_register("chart_conversational_argmax", ChartConversationalArgmaxQA, "diagram")
from .chart_conversational_year_qa import ChartConversationalYearQA
_register("chart_conversational_year", ChartConversationalYearQA, "diagram")
from .chart_counterfactual_pattern_qa import ChartCounterfactualPatternQA
_register("chart_counterfactual_pattern", ChartCounterfactualPatternQA, "diagram")
from .chart_dialogue_arithmetic_qa import ChartDialogueArithmeticQA
_register("chart_dialogue_arithmetic", ChartDialogueArithmeticQA, "diagram")
from .chart_fact_checking_qa import ChartFactCheckingQA
_register("chart_fact_checking", ChartFactCheckingQA, "diagram")
from .chart_filter_aggregate_qa import ChartFilterAggregateQA
_register("chart_filter_aggregate", ChartFilterAggregateQA, "diagram")
from .chart_hypothetical_modify_qa import ChartHypotheticalModifyQA
_register("chart_hypothetical_modify", ChartHypotheticalModifyQA, "diagram")
from .chart_hypothetical_qa import ChartHypotheticalQA
_register("chart_hypothetical", ChartHypotheticalQA, "diagram")
from .chart_kth_largest_qa import ChartKthLargestQA
_register("chart_kth_largest", ChartKthLargestQA, "diagram")
from .chart_linear_forecast_qa import ChartLinearForecastQA
_register("chart_linear_forecast", ChartLinearForecastQA, "diagram")
from .chart_mcq_letter_only_qa import ChartMcqLetterOnlyQA
_register("chart_mcq_letter_only", ChartMcqLetterOnlyQA, "diagram")
from .chart_multistep_qa import ChartMultistepQA
_register("chart_multistep", ChartMultistepQA, "diagram")
from .chart_no_labels_qa import ChartNoLabelsQA
_register("chart_no_labels", ChartNoLabelsQA, "diagram")
from .chart_percent_change_qa import ChartPercentChangeQA
_register("chart_percent_change", ChartPercentChangeQA, "diagram")
from .chart_quantifier_claim_qa import ChartQuantifierClaimQA
_register("chart_quantifier_claim", ChartQuantifierClaimQA, "diagram")
from .chart_set_membership_claim_qa import ChartSetMembershipClaimQA
_register("chart_set_membership_claim", ChartSetMembershipClaimQA, "diagram")
from .chart_threshold_filter_count_qa import ChartThresholdFilterCountQA
_register("chart_threshold_filter_count", ChartThresholdFilterCountQA, "diagram")
from .chart_unanswerable_trap_qa import ChartUnanswerableTrapQA
_register("chart_unanswerable_trap", ChartUnanswerableTrapQA, "diagram")
from .chart_with_context_paragraph_qa import ChartWithContextParagraphQA
_register("chart_with_context_paragraph", ChartWithContextParagraphQA, "diagram")
from .chart_with_latex_label_qa import ChartWithLatexLabelQA
_register("chart_with_latex_label", ChartWithLatexLabelQA, "diagram")
from .chart_yesno_threshold_qa import ChartYesnoThresholdQA
_register("chart_yesno_threshold", ChartYesnoThresholdQA, "diagram")
from .chiral_object_identification_qa import ChiralObjectIdentificationQA
_register("chiral_object_identification", ChiralObjectIdentificationQA, "spatial")
from .chirality_pair_discrimination_qa import ChiralityPairDiscriminationQA
_register("chirality_pair_discrimination", ChiralityPairDiscriminationQA, "spatial")
from .chromatic_index_qa import ChromaticIndexQA
_register("chromatic_index", ChromaticIndexQA, "diagram")
from .circle_inscribed_angle_warmup_qa import CircleInscribedAngleWarmupQA
_register("circle_inscribed_angle_warmup", CircleInscribedAngleWarmupQA, "math")
from .circle_size_number_relation_qa import CircleSizeNumberRelationQA
_register("circle_size_number_relation", CircleSizeNumberRelationQA, "math")
from .circle_theorem_qa import CircleTheoremQA
_register("circle_theorem", CircleTheoremQA, "math")
from .circuit_logic_qa import CircuitLogicQA
_register("circuit_logic", CircuitLogicQA, "diagram")
from .circuit_output_prediction_qa import CircuitOutputPredictionQA
_register("circuit_output_prediction", CircuitOutputPredictionQA, "diagram")
from .clock_angle_qa import ClockAngleQA
_register("clock_angle", ClockAngleQA, "math")
from .clock_elapsed_minutes_qa import ClockElapsedMinutesQA
_register("clock_elapsed_minutes", ClockElapsedMinutesQA, "math")
from .code_deduction_qa import CodeDeductionQA
_register("code_deduction", CodeDeductionQA, "pattern")
from .color_grid_pattern_qa import ColorGridPatternQA
_register("color_grid_pattern", ColorGridPatternQA, "pattern")
from .colored_grid_rotation_qa import ColoredGridRotationQA
_register("colored_grid_rotation", ColoredGridRotationQA, "spatial")
from .combinatorial_geometry_configs_qa import CombinatorialGeometryConfigsQA
_register("combinatorial_geometry_configs", CombinatorialGeometryConfigsQA, "math")
from .combinatorics_visual_qa import CombinatoricsVisualQA
_register("combinatorics_visual", CombinatoricsVisualQA, "pattern")
from .compass_premise_pair_qa import CompassPremisePairQA
_register("compass_premise_pair", CompassPremisePairQA, "spatial")
from .composite_2step_geometry_qa import Composite2StepGeometryQA
_register("composite_2step_geometry", Composite2StepGeometryQA, "math")
from .composite_3d_volume_qa import Composite3DVolumeQA
_register("composite_3d_volume", Composite3DVolumeQA, "spatial")
from .composite_3step_cylinder_qa import Composite3StepCylinderQA
_register("composite_3step_cylinder", Composite3StepCylinderQA, "math")
from .composite_3step_transformation_qa import Composite3StepTransformationQA
_register("composite_3step_transformation", Composite3StepTransformationQA, "math")
from .composite_3step_trapezoid_qa import Composite3StepTrapezoidQA
_register("composite_3step_trapezoid", Composite3StepTrapezoidQA, "math")
from .composite_area_decomposition_qa import CompositeAreaDecompositionQA
_register("composite_area_decomposition", CompositeAreaDecompositionQA, "math")
from .composite_perimeter_qa import CompositePerimeterQA
_register("composite_perimeter", CompositePerimeterQA, "math")
from .compound_3d_volume_qa import Compound3dVolumeQA
_register("compound_3d_volume", Compound3dVolumeQA, "spatial")
from .conditional_logic_chain_qa import ConditionalLogicChainQA
_register("conditional_logic_chain", ConditionalLogicChainQA, "pattern")
from .conditional_probability_visual_qa import ConditionalProbabilityVisualQA
_register("conditional_probability_visual", ConditionalProbabilityVisualQA, "math")
from .cone_to_cylinder_pour_qa import ConeToCylinderPourQA
_register("cone_to_cylinder_pour", ConeToCylinderPourQA, "math")
from .cone_volume_from_diagram_qa import ConeVolumeFromDiagramQA
_register("cone_volume_from_diagram", ConeVolumeFromDiagramQA, "math")
from .confusion_matrix_diagonal_qa import ConfusionMatrixDiagonalQA
_register("confusion_matrix_diagonal", ConfusionMatrixDiagonalQA, "diagram")
from .congruence_proof_qa import CongruenceProofQA
_register("congruence_proof", CongruenceProofQA, "math")
from .conic_eccentricity_mcq_qa import ConicEccentricityMCQQA
_register("conic_eccentricity_mcq", ConicEccentricityMCQQA, "math")
from .conic_identification_qa import ConicIdentificationQA
_register("conic_identification", ConicIdentificationQA, "math")
from .container_max_water_qa import ContainerMaxWaterQA
_register("container_max_water", ContainerMaxWaterQA, "pattern")
from .continuity_at_point_qa import ContinuityAtPointQA
_register("continuity_at_point", ContinuityAtPointQA, "math")
from .contour_region_area_qa import ContourRegionAreaQA
_register("contour_region_area", ContourRegionAreaQA, "math")
from .coord_transform_algebraic_qa import CoordTransformAlgebraicQA
_register("coord_transform_algebraic", CoordTransformAlgebraicQA, "math")
from .coordinate_geometry_algebra_qa import CoordinateGeometryAlgebraQA
_register("coordinate_geometry_algebra", CoordinateGeometryAlgebraQA, "math")
from .coordinate_geometry_named_point_qa import CoordinateGeometryNamedPointQA
_register("coordinate_geometry_named_point", CoordinateGeometryNamedPointQA, "math")
from .coordinate_geometry_qa import CoordinateGeometryQA
_register("coordinate_geometry", CoordinateGeometryQA, "math")
from .coordinate_transform_qa import CoordinateTransformQA
_register("coordinate_transform", CoordinateTransformQA, "math")
from .cross_reference_chart_qa import CrossReferenceChartQA
_register("cross_reference_chart", CrossReferenceChartQA, "diagram")
from .cross_section_qa import CrossSectionQA
_register("cross_section", CrossSectionQA, "spatial")
from .cryptarithm_qa import CryptarithmQA
_register("cryptarithm", CryptarithmQA, "pattern")
from .cube_assembly_missing_piece_qa import CubeAssemblyMissingPieceQA
_register("cube_assembly_missing_piece", CubeAssemblyMissingPieceQA, "spatial")
from .cube_assembly_qa import CubeAssemblyQA
_register("cube_assembly", CubeAssemblyQA, "spatial")
from .cube_decomposition_count_qa import CubeDecompositionCountQA
_register("cube_decomposition_count", CubeDecompositionCountQA, "spatial")
from .cube_decomposition_id_qa import CubeDecompositionIdQA
_register("cube_decomposition_id", CubeDecompositionIdQA, "spatial")
from .cube_face_opposite_pick_qa import CubeFaceOppositePickQA
_register("cube_face_opposite_pick", CubeFaceOppositePickQA, "spatial")
from .cube_net_fold_pick_qa import CubeNetFoldPickQA
_register("cube_net_fold_pick", CubeNetFoldPickQA, "spatial")
from .cube_perspective_swap_qa import CubePerspectiveSwapQA
_register("cube_perspective_swap", CubePerspectiveSwapQA, "spatial")
from .cuboid_into_cubes_qa import CuboidIntoCubesQA
_register("cuboid_into_cubes", CuboidIntoCubesQA, "spatial")
from .curve_functional_form_qa import CurveFunctionalFormQA
_register("curve_functional_form", CurveFunctionalFormQA, "math")
from .curve_property_identify_qa import CurvePropertyIdentifyQA
_register("curve_property_identify", CurvePropertyIdentifyQA, "diagram")
from .cut_rearrange_shape_qa import CutRearrangeShapeQA
_register("cut_rearrange_shape", CutRearrangeShapeQA, "spatial")
from .cycle_diagram_qa import CycleDiagramQA
_register("cycle_diagram", CycleDiagramQA, "diagram")
from .cyclic_group_fill_qa import CyclicGroupFillQA
_register("cyclic_group_fill", CyclicGroupFillQA, "pattern")
from .cyclic_quadrilateral_advanced_qa import CyclicQuadrilateralAdvancedQA
_register("cyclic_quadrilateral_advanced", CyclicQuadrilateralAdvancedQA, "math")
from .cylinder_surface_area_net_qa import CylinderSurfaceAreaNetQA
_register("cylinder_surface_area_net", CylinderSurfaceAreaNetQA, "math")
from .dartboard_score_qa import DartboardScoreQA
_register("dartboard_score", DartboardScoreQA, "count")
from .dashboard_conversational_qa import DashboardConversationalQA
_register("dashboard_conversational", DashboardConversationalQA, "diagram")
from .dense_2d_count_warmup_qa import Dense2DCountWarmupQA
_register("dense_2d_count_warmup", Dense2DCountWarmupQA, "count")
from .density_plot_peak_qa import DensityPlotPeakQA
_register("density_plot_peak", DensityPlotPeakQA, "diagram")
from .dependency_graph_qa import DependencyGraphQA
_register("dependency_graph", DependencyGraphQA, "diagram")
from .depth_order_qa import DepthOrderQA
_register("depth_order", DepthOrderQA, "spatial")
from .derivative_graph_qa import DerivativeGraphQA
_register("derivative_graph", DerivativeGraphQA, "diagram")
from .differentiable_at_point_qa import DifferentiableAtPointQA
_register("differentiable_at_point", DifferentiableAtPointQA, "math")
from .digit_permutation_max_qa import DigitPermutationMaxQA
_register("digit_permutation_max", DigitPermutationMaxQA, "pattern")
from .directed_graph_qa import DirectedGraphQA
_register("directed_graph", DirectedGraphQA, "diagram")
from .distance_between_points_qa import DistanceBetweenPointsQA
_register("distance_between_points", DistanceBetweenPointsQA, "math")
from .diverging_bar_qa import DivergingBarQA
_register("diverging_bar", DivergingBarQA, "diagram")
from .donut_chart_qa import DonutChartQA
_register("donut_chart", DonutChartQA, "diagram")
from .dual_axis_chart_qa import DualAxisChartQA
_register("dual_axis_chart", DualAxisChartQA, "diagram")
from .edge_sum_graph_qa import EdgeSumGraphQA
_register("edge_sum_graph", EdgeSumGraphQA, "diagram")
from .edges_removable_count_qa import EdgesRemovableCountQA
_register("edges_removable_count", EdgesRemovableCountQA, "count")
from .ego_motion_compound_qa import EgoMotionCompoundQA
_register("ego_motion_compound", EgoMotionCompoundQA, "spatial")
from .energy_level_qa import EnergyLevelQA
_register("energy_level", EnergyLevelQA, "diagram")
from .equation_blank_fill_qa import EquationBlankFillQA
_register("equation_blank_fill", EquationBlankFillQA, "math")
from .error_bar_chart_qa import ErrorBarChartQA
_register("error_bar_chart", ErrorBarChartQA, "diagram")
from .eulerian_cycle_find_qa import EulerianCycleFindQA
_register("eulerian_cycle_find", EulerianCycleFindQA, "diagram")
from .eulerian_path_count_qa import EulerianPathCountQA
_register("eulerian_path_count", EulerianPathCountQA, "count")
from .eulerian_path_find_qa import EulerianPathFindQA
_register("eulerian_path_find", EulerianPathFindQA, "diagram")
from .eulero_grid_qa import EuleroGridQA
_register("eulero_grid", EuleroGridQA, "spatial")
from .expression_substitute_evaluate_qa import ExpressionSubstituteEvaluateQA
_register("expression_substitute_evaluate", ExpressionSubstituteEvaluateQA, "math")
from .exterior_angle_chain_qa import ExteriorAngleChainQA
_register("exterior_angle_chain", ExteriorAngleChainQA, "math")
from .feature_counting_classification_qa import FeatureCountingClassificationQA
_register("feature_counting_classification", FeatureCountingClassificationQA, "count")
from .figure_counting_qa import FigureCountingQA
_register("figure_counting", FigureCountingQA, "count")
from .figure_sequence_next_qa import FigureSequenceNextQA
_register("figure_sequence_next", FigureSequenceNextQA, "pattern")
from .figure_to_expression_qa import FigureToExpressionQA
_register("figure_to_expression", FigureToExpressionQA, "diagram")
from .figure_to_quantity_table_qa import FigureToQuantityTableQA
_register("figure_to_quantity_table", FigureToQuantityTableQA, "count")
from .flowchart_qa import FlowchartQA
_register("flowchart", FlowchartQA, "diagram")
from .fold_constraint_solve_qa import FoldConstraintSolveQA
_register("fold_constraint_solve", FoldConstraintSolveQA, "pattern")
from .food_web_qa import FoodWebQA
_register("food_web", FoodWebQA, "diagram")
from .fractal_pattern_qa import FractalPatternQA
_register("fractal_pattern", FractalPatternQA, "pattern")
from .frequency_table_qa import FrequencyTableQA
_register("frequency_table", FrequencyTableQA, "diagram")
from .function_formula_from_graph_qa import FunctionFormulaFromGraphQA
_register("function_formula_from_graph", FunctionFormulaFromGraphQA, "diagram")
from .function_graph_value_read_qa import FunctionGraphValueReadQA
_register("function_graph_value_read", FunctionGraphValueReadQA, "diagram")
from .function_inj_surj_judge_qa import FunctionInjSurjJudgeQA
_register("function_inj_surj_judge", FunctionInjSurjJudgeQA, "pattern")
from .function_peak_count_qa import FunctionPeakCountQA
_register("function_peak_count", FunctionPeakCountQA, "count")
from .function_periodic_judge_qa import FunctionPeriodicJudgeQA
_register("function_periodic_judge", FunctionPeriodicJudgeQA, "pattern")
from .function_region_area_compose_qa import FunctionRegionAreaComposeQA
_register("function_region_area_compose", FunctionRegionAreaComposeQA, "math")
from .function_threshold_crossing_count_qa import FunctionThresholdCrossingCountQA
_register("function_threshold_crossing_count", FunctionThresholdCrossingCountQA, "count")
from .futoshiki_solve_qa import FutoshikiSolveQA
_register("futoshiki_solve", FutoshikiSolveQA, "pattern")
from .gantt_chart_qa import GanttChartQA
_register("gantt_chart", GanttChartQA, "diagram")
from .gear_rotation_direction_qa import GearRotationDirectionQA
_register("gear_rotation_direction", GearRotationDirectionQA, "spatial")
from .gear_train_qa import GearTrainQA
_register("gear_train", GearTrainQA, "spatial")
from .geometric_transform_qa import GeometricTransformQA
_register("geometric_transform", GeometricTransformQA, "math")
from .geometry_label_reading_qa import GeometryLabelReadingQA
_register("geometry_label_reading", GeometryLabelReadingQA, "math")
from .graph_coloring_qa import GraphColoringQA
_register("graph_coloring", GraphColoringQA, "diagram")
from .graph_connectivity_qa import GraphConnectivityQA
_register("graph_connectivity", GraphConnectivityQA, "diagram")
from .graph_enumeration_from_image_qa import GraphEnumerationFromImageQA
_register("graph_enumeration_from_image", GraphEnumerationFromImageQA, "diagram")
from .graph_isomorphism_qa import GraphIsomorphismQA
_register("graph_isomorphism", GraphIsomorphismQA, "diagram")
from .graph_properties_qa import GraphPropertiesQA
_register("graph_properties", GraphPropertiesQA, "diagram")
from .grid_cell_count_with_rules_qa import GridCellCountWithRulesQA
_register("grid_cell_count_with_rules", GridCellCountWithRulesQA, "count")
from .grid_navigation_with_obstacles_qa import GridNavigationWithObstaclesQA
_register("grid_navigation_with_obstacles", GridNavigationWithObstaclesQA, "spatial")
from .grid_puzzle_qa import GridPuzzleQA
_register("grid_puzzle", GridPuzzleQA, "pattern")
from .h_index_qa import HIndexQA
_register("h_index", HIndexQA, "diagram")
from .hamiltonian_circuit_judge_qa import HamiltonianCircuitJudgeQA
_register("hamiltonian_circuit_judge", HamiltonianCircuitJudgeQA, "diagram")
from .hamiltonian_cycle_find_qa import HamiltonianCycleFindQA
_register("hamiltonian_cycle_find", HamiltonianCycleFindQA, "diagram")
from .hamiltonian_path_find_qa import HamiltonianPathFindQA
_register("hamiltonian_path_find", HamiltonianPathFindQA, "diagram")
from .handwritten_expression_qa import HandwrittenExpressionQA
_register("handwritten_expression", HandwrittenExpressionQA, "diagram")
from .hashi_bridges_qa import HashiBridgesQA
_register("hashi_bridges", HashiBridgesQA, "pattern")
from .heatmap_advanced_qa import HeatmapAdvancedQA
_register("heatmap_advanced", HeatmapAdvancedQA, "diagram")
from .heatmap_pattern_identification_qa import HeatmapPatternIdentificationQA
_register("heatmap_pattern_identification", HeatmapPatternIdentificationQA, "diagram")
from .heatmap_row_condition_count_qa import HeatmapRowConditionCountQA
_register("heatmap_row_condition_count", HeatmapRowConditionCountQA, "diagram")
from .hexagon_size_comparison_qa import HexagonSizeComparisonQA
_register("hexagon_size_comparison", HexagonSizeComparisonQA, "pattern")
from .hexagon_triple_relation_qa import HexagonTripleRelationQA
_register("hexagon_triple_relation", HexagonTripleRelationQA, "pattern")
from .hidden_cube_inference_qa import HiddenCubeInferenceQA
_register("hidden_cube_inference", HiddenCubeInferenceQA, "spatial")
from .hills_valleys_count_qa import HillsValleysCountQA
_register("hills_valleys_count", HillsValleysCountQA, "count")
from .histogram_qa import HistogramQA
_register("histogram", HistogramQA, "diagram")
from .hitori_solve_qa import HitoriSolveQA
_register("hitori_solve", HitoriSolveQA, "pattern")
from .hyperbola_find_k_qa import HyperbolaFindKQA
_register("hyperbola_find_k", HyperbolaFindKQA, "math")
from .hyperbola_rectangle_area_qa import HyperbolaRectangleAreaQA
_register("hyperbola_rectangle_area", HyperbolaRectangleAreaQA, "math")
from .image_rotation_match_qa import ImageRotationMatchQA
_register("image_rotation_match", ImageRotationMatchQA, "spatial")
from .implicit_function_level_set_qa import ImplicitFunctionLevelSetQA
_register("implicit_function_level_set", ImplicitFunctionLevelSetQA, "math")
from .indoor_region_compass_qa import IndoorRegionCompassQA
_register("indoor_region_compass", IndoorRegionCompassQA, "spatial")
from .inductive_rule_discovery_qa import InductiveRuleDiscoveryQA
_register("inductive_rule_discovery", InductiveRuleDiscoveryQA, "pattern")
from .inequality_system_qa import InequalitySystemQA
_register("inequality_system", InequalitySystemQA, "math")
from .infographic_business_qa import InfographicBusinessQA
_register("infographic_business", InfographicBusinessQA, "diagram")
from .infographic_composite_qa import InfographicCompositeQA
_register("infographic_composite", InfographicCompositeQA, "diagram")
from .inscribed_polygon_qa import InscribedPolygonQA
_register("inscribed_polygon", InscribedPolygonQA, "math")
from .instrument_panel_qa import InstrumentPanelQA
_register("instrument_panel", InstrumentPanelQA, "diagram")
from .inverse_function_qa import InverseFunctionQA
_register("inverse_function", InverseFunctionQA, "math")
from .iq_compass_movement_qa import IQCompassMovementQA
_register("iq_compass_movement", IQCompassMovementQA, "spatial")
from .iq_multi_answer_pattern_qa import IQMultiAnswerPatternQA
_register("iq_multi_answer_pattern", IQMultiAnswerPatternQA, "pattern")
from .iq_series_mixed_rules_qa import IQSeriesMixedRulesQA
_register("iq_series_mixed_rules", IQSeriesMixedRulesQA, "pattern")
from .isometric_counting_qa import IsometricCountingQA
_register("isometric_counting", IsometricCountingQA, "spatial")
from .jigsaw_labelled_edge_qa import JigsawLabelledEdgeQA
_register("jigsaw_labelled_edge", JigsawLabelledEdgeQA, "pattern")
from .jigsaw_piece_match_qa import JigsawPieceMatchQA
_register("jigsaw_piece_match", JigsawPieceMatchQA, "pattern")
from .kakuro_qa import KakuroQA
_register("kakuro", KakuroQA, "pattern")
from .kepler_orbit_speed_qa import KeplerOrbitSpeedQA
_register("kepler_orbit_speed", KeplerOrbitSpeedQA, "math")
from .kinematics_graph_qa import KinematicsGraphQA
_register("kinematics_graph", KinematicsGraphQA, "diagram")
from .kruskal_first_edge_qa import KruskalFirstEdgeQA
_register("kruskal_first_edge", KruskalFirstEdgeQA, "diagram")
from .kukurasu_solve_qa import KukurasuSolveQA
_register("kukurasu_solve", KukurasuSolveQA, "pattern")
from .labeled_parts_diagram_qa import LabeledPartsDiagramQA
_register("labeled_parts_diagram", LabeledPartsDiagramQA, "diagram")
from .largest_rectangle_in_histogram_qa import LargestRectangleInHistogramQA
_register("largest_rectangle_in_histogram", LargestRectangleInHistogramQA, "diagram")
from .latin_square_count_qa import LatinSquareCountQA
_register("latin_square_count", LatinSquareCountQA, "count")
from .latin_square_fill_qa import LatinSquareFillQA
_register("latin_square_fill", LatinSquareFillQA, "pattern")
from .lattice_in_disk_count_qa import LatticeInDiskCountQA
_register("lattice_in_disk_count", LatticeInDiskCountQA, "count")
from .lattice_path_count_qa import LatticePathCountQA
_register("lattice_path_count", LatticePathCountQA, "spatial")
from .lattice_path_qa import LatticePathQA
_register("lattice_path", LatticePathQA, "spatial")
from .layer_by_layer_count_qa import LayerByLayerCountQA
_register("layer_by_layer_count", LayerByLayerCountQA, "count")
from .layered_stack_count_qa import LayeredStackCountQA
_register("layered_stack_count", LayeredStackCountQA, "spatial")
from .lever_pulley_qa import LeverPulleyQA
_register("lever_pulley", LeverPulleyQA, "math")
from .line_axes_triangle_area_qa import LineAxesTriangleAreaQA
_register("line_axes_triangle_area", LineAxesTriangleAreaQA, "diagram")
from .line_chart_first_crossing_year_qa import LineChartFirstCrossingYearQA
_register("line_chart_first_crossing_year", LineChartFirstCrossingYearQA, "diagram")
from .line_chart_qa import LineChartQA
_register("line_chart", LineChartQA, "diagram")
from .line_chart_threshold_crossing_count_qa import LineChartThresholdCrossingCountQA
_register("line_chart_threshold_crossing_count", LineChartThresholdCrossingCountQA, "diagram")
from .line_chart_trend_word_qa import LineChartTrendWordQA
_register("line_chart_trend_word", LineChartTrendWordQA, "diagram")
from .line_fixed_point_family_qa import LineFixedPointFamilyQA
_register("line_fixed_point_family", LineFixedPointFamilyQA, "math")
from .line_hyperbola_intersection_qa import LineHyperbolaIntersectionQA
_register("line_hyperbola_intersection", LineHyperbolaIntersectionQA, "math")
from .line_parallel_perp_qa import LineParallelPerpQA
_register("line_parallel_perp", LineParallelPerpQA, "math")
from .line_quadrant_judge_qa import LineQuadrantJudgeQA
_register("line_quadrant_judge", LineQuadrantJudgeQA, "math")
from .line_slope_intercept_qa import LineSlopeInterceptQA
_register("line_slope_intercept", LineSlopeInterceptQA, "math")
from .lis_length_qa import LISLengthQA
_register("lis_length", LISLengthQA, "pattern")
from .logic_grid_qa import LogicGridQA
_register("logic_grid", LogicGridQA, "spatial")
from .logical_negation_chain_qa import LogicalNegationChainQA
_register("logical_negation_chain", LogicalNegationChainQA, "pattern")
from .magic_cross_qa import MagicCrossQA
_register("magic_cross", MagicCrossQA, "pattern")
from .magic_lines_grid_qa import MagicLinesGridQA
_register("magic_lines_grid", MagicLinesGridQA, "spatial")
from .map_distance_qa import MapDistanceQA
_register("map_distance", MapDistanceQA, "spatial")
from .map_route_optimization_qa import MapRouteOptimizationQA
_register("map_route_optimization", MapRouteOptimizationQA, "spatial")
from .marked_corner_after_rotation_qa import MarkedCornerAfterRotationQA
_register("marked_corner_after_rotation", MarkedCornerAfterRotationQA, "spatial")
from .markov_transition_qa import MarkovTransitionQA
_register("markov_transition", MarkovTransitionQA, "diagram")
from .matchstick_geometric_removal_qa import MatchstickGeometricRemovalQA
_register("matchstick_geometric_removal", MatchstickGeometricRemovalQA, "pattern")
from .matchstick_removal_qa import MatchstickRemovalQA
_register("matchstick_removal", MatchstickRemovalQA, "pattern")
from .matrix_completion_5x5_qa import MatrixCompletion5x5QA
_register("matrix_completion_5x5", MatrixCompletion5x5QA, "pattern")
from .matrix_operation_qa import MatrixOperationQA
_register("matrix_operation", MatrixOperationQA, "math")
from .matrix_pattern_qa import MatrixPatternQA
_register("matrix_pattern", MatrixPatternQA, "pattern")
from .max_flow_dag_qa import MaxFlowDAGQA
_register("max_flow_dag", MaxFlowDAGQA, "diagram")
from .maze_pathfind_qa import MazePathfindQA
_register("maze_pathfind", MazePathfindQA, "spatial")
from .maze_solution_length_qa import MazeSolutionLengthQA
_register("maze_solution_length", MazeSolutionLengthQA, "spatial")
from .maze_turn_sequence_qa import MazeTurnSequenceQA
_register("maze_turn_sequence", MazeTurnSequenceQA, "spatial")
from .mcq_answer_not_in_options_qa import MCQAnswerNotInOptionsQA
_register("mcq_answer_not_in_options", MCQAnswerNotInOptionsQA, "pattern")
from .medication_deduction_qa import MedicationDeductionQA
_register("medication_deduction", MedicationDeductionQA, "pattern")
from .minesweeper_solve_qa import MinesweeperSolveQA
_register("minesweeper_solve", MinesweeperSolveQA, "pattern")
from .mirror_plane_identification_qa import MirrorPlaneIdentificationQA
_register("mirror_plane_identification", MirrorPlaneIdentificationQA, "spatial")
from .mirror_reflection_qa import MirrorReflectionQA
_register("mirror_reflection", MirrorReflectionQA, "spatial")
from .mirror_text_inverse_qa import MirrorTextInverseQA
_register("mirror_text_inverse", MirrorTextInverseQA, "spatial")
from .mirror_vs_rotation_discriminator_qa import MirrorVsRotationDiscriminatorQA
_register("mirror_vs_rotation_discriminator", MirrorVsRotationDiscriminatorQA, "spatial")
from .missing_grid_count_qa import MissingGridCountQA
_register("missing_grid_count", MissingGridCountQA, "count")
from .multi_attribute_partition_qa import MultiAttributePartitionQA
_register("multi_attribute_partition", MultiAttributePartitionQA, "pattern")
from .multi_axis_chart_qa import MultiAxisChartQA
_register("multi_axis_chart", MultiAxisChartQA, "diagram")
from .multi_axis_rotation_chain_qa import MultiAxisRotationChainQA
_register("multi_axis_rotation_chain", MultiAxisRotationChainQA, "spatial")
from .multi_chart_qa import MultiChartQA
_register("multi_chart", MultiChartQA, "diagram")
from .multi_hop_metric_chain_qa import MultiHopMetricChainQA
_register("multi_hop_metric_chain", MultiHopMetricChainQA, "pattern")
from .multi_line_dominance_claim_qa import MultiLineDominanceClaimQA
_register("multi_line_dominance_claim", MultiLineDominanceClaimQA, "diagram")
from .multi_line_slope_compare_qa import MultiLineSlopeCompareQA
_register("multi_line_slope_compare", MultiLineSlopeCompareQA, "diagram")
from .multi_premise_deduction_qa import MultiPremiseDeductionQA
_register("multi_premise_deduction", MultiPremiseDeductionQA, "pattern")
from .multi_table_join_qa import MultiTableJoinQA
_register("multi_table_join", MultiTableJoinQA, "diagram")
from .multi_triangle_length_chain_qa import MultiTriangleLengthChainQA
_register("multi_triangle_length_chain", MultiTriangleLengthChainQA, "math")
from .multi_view_3d_reconstruction_qa import MultiView3dReconstructionQA
_register("multi_view_3d_reconstruction", MultiView3dReconstructionQA, "spatial")
from .multi_view_consistency_check_qa import MultiViewConsistencyCheckQA
_register("multi_view_consistency_check", MultiViewConsistencyCheckQA, "spatial")
from .multi_view_cube_count_qa import MultiViewCubeCountQA
_register("multi_view_cube_count", MultiViewCubeCountQA, "spatial")
from .multi_view_object_match_qa import MultiViewObjectMatchQA
_register("multi_view_object_match", MultiViewObjectMatchQA, "spatial")
from .near_far_mcq_qa import NearFarMcqQA
_register("near_far_mcq", NearFarMcqQA, "spatial")
from .net_folding_qa import NetFoldingQA
_register("net_folding", NetFoldingQA, "spatial")
from .net_validity_advanced_qa import NetValidityAdvancedQA
_register("net_validity_advanced", NetValidityAdvancedQA, "pattern")
from .network_topology_qa import NetworkTopologyQA
_register("network_topology", NetworkTopologyQA, "diagram")
from .nibbles_snake_game_qa import NibblesSnakeGameQA
_register("nibbles_snake_game", NibblesSnakeGameQA, "pattern")
from .nonogram_solve_qa import NonogramSolveQA
_register("nonogram_solve", NonogramSolveQA, "pattern")
from .number_line_qa import NumberLineQA
_register("number_line", NumberLineQA, "diagram")
from .number_tree_inverse_qa import NumberTreeInverseQA
_register("number_tree_inverse", NumberTreeInverseQA, "diagram")
from .numbrix_solve_qa import NumbrixSolveQA
_register("numbrix_solve", NumbrixSolveQA, "pattern")
from .numeric_commonsense_visual_qa import NumericCommonsenseVisualQA
_register("numeric_commonsense_visual", NumericCommonsenseVisualQA, "spatial")
from .numerical_pyramid_qa import NumericalPyramidQA
_register("numerical_pyramid", NumericalPyramidQA, "math")
from .object_count_with_occlusion_qa import ObjectCountWithOcclusionQA
_register("object_count_with_occlusion", ObjectCountWithOcclusionQA, "spatial")
from .object_tracking_across_frames_qa import ObjectTrackingAcrossFramesQA
_register("object_tracking_across_frames", ObjectTrackingAcrossFramesQA, "spatial")
from .object_vertical_relation_qa import ObjectVerticalRelationQA
_register("object_vertical_relation", ObjectVerticalRelationQA, "spatial")
from .odd_one_out_qa import OddOneOutQA
_register("odd_one_out", OddOneOutQA, "pattern")
from .omitted_operator_qa import OmittedOperatorQA
_register("omitted_operator", OmittedOperatorQA, "pattern")
from .open_top_container_sa_qa import OpenTopContainerSaQA
_register("open_top_container_sa", OpenTopContainerSaQA, "pattern")
from .optical_illusion_lines_qa import OpticalIllusionLinesQA
_register("optical_illusion_lines", OpticalIllusionLinesQA, "spatial")
from .optics_diagram_qa import OpticsDiagramQA
_register("optics_diagram", OpticsDiagramQA, "spatial")
from .orthographic_projection_qa import OrthographicProjectionQA
_register("orthographic_projection", OrthographicProjectionQA, "spatial")
from .overlapping_shape_count_qa import OverlappingShapeCountQA
_register("overlapping_shape_count", OverlappingShapeCountQA, "count")
from .paired_count_raven_qa import PairedCountRavenQA
_register("paired_count_raven", PairedCountRavenQA, "pattern")
from .paper_fold_crease_length_qa import PaperFoldCreaseLengthQA
_register("paper_fold_crease_length", PaperFoldCreaseLengthQA, "spatial")
from .paper_folding_qa import PaperFoldingQA
_register("paper_folding", PaperFoldingQA, "spatial")
from .parabola_3point_fit_qa import Parabola3PointFitQA
_register("parabola_3point_fit", Parabola3PointFitQA, "math")
from .parabola_line_intersection_qa import ParabolaLineIntersectionQA
_register("parabola_line_intersection", ParabolaLineIntersectionQA, "math")
from .parabola_roots_count_translate_qa import ParabolaRootsCountTranslateQA
_register("parabola_roots_count_translate", ParabolaRootsCountTranslateQA, "count")
from .parabola_sign_inference_qa import ParabolaSignInferenceQA
_register("parabola_sign_inference", ParabolaSignInferenceQA, "math")
from .parabola_symmetric_point_qa import ParabolaSymmetricPointQA
_register("parabola_symmetric_point", ParabolaSymmetricPointQA, "math")
from .parabola_translate_vertex_form_qa import ParabolaTranslateVertexFormQA
_register("parabola_translate_vertex_form", ParabolaTranslateVertexFormQA, "math")
from .parabola_vertex_qa import ParabolaVertexQA
_register("parabola_vertex", ParabolaVertexQA, "math")
from .parabola_vieta_qa import ParabolaVietaQA
_register("parabola_vieta", ParabolaVietaQA, "math")
from .parallel_lines_angle_chain_qa import ParallelLinesAngleChainQA
_register("parallel_lines_angle_chain", ParallelLinesAngleChainQA, "math")
from .parallel_rhombus_property_qa import ParallelRhombusPropertyQA
_register("parallel_rhombus_property", ParallelRhombusPropertyQA, "math")
from .parallel_transversal_angles_qa import ParallelTransversalAnglesQA
_register("parallel_transversal_angles", ParallelTransversalAnglesQA, "math")
from .parametric_curve_point_qa import ParametricCurvePointQA
_register("parametric_curve_point", ParametricCurvePointQA, "math")
from .partial_occluded_cube_count_qa import PartialOccludedCubeCountQA
_register("partial_occluded_cube_count", PartialOccludedCubeCountQA, "spatial")
from .passage_factual_recall_qa import PassageFactualRecallQA
_register("passage_factual_recall", PassageFactualRecallQA, "diagram")
from .path_counting_qa import PathCountingQA
_register("path_counting", PathCountingQA, "count")
from .path_length_grid_count_qa import PathLengthGridCountQA
_register("path_length_grid_count", PathLengthGridCountQA, "count")
from .pattern_rule_multi_example_qa import PatternRuleMultiExampleQA
_register("pattern_rule_multi_example", PatternRuleMultiExampleQA, "pattern")
from .pendulum_compare_qa import PendulumCompareQA
_register("pendulum_compare", PendulumCompareQA, "math")
from .periodic_table_qa import PeriodicTableQA
_register("periodic_table", PeriodicTableQA, "diagram")
from .perspective_shift_qa import PerspectiveShiftQA
_register("perspective_shift", PerspectiveShiftQA, "spatial")
from .phase_diagram_qa import PhaseDiagramQA
_register("phase_diagram", PhaseDiagramQA, "diagram")
from .photo_chart_reading_qa import PhotoChartReadingQA
_register("photo_chart_reading", PhotoChartReadingQA, "diagram")
from .photo_clock_dial_qa import PhotoClockDialQA
_register("photo_clock_dial", PhotoClockDialQA, "math")
from .photo_measurement_estimate_qa import PhotoMeasurementEstimateQA
_register("photo_measurement_estimate", PhotoMeasurementEstimateQA, "count")
from .physical_setup_length_qa import PhysicalSetupLengthQA
_register("physical_setup_length", PhysicalSetupLengthQA, "math")
from .pictogram_rule_qa import PictogramRuleQA
_register("pictogram_rule", PictogramRuleQA, "diagram")
from .pie_chart_advanced_qa import PieChartAdvancedQA
_register("pie_chart_advanced", PieChartAdvancedQA, "diagram")
from .pie_chart_percentage_qa import PieChartPercentageQA
_register("pie_chart_percentage", PieChartPercentageQA, "diagram")
from .pie_chart_rebalance_qa import PieChartRebalanceQA
_register("pie_chart_rebalance", PieChartRebalanceQA, "diagram")
from .piecewise_domain_identification_qa import PiecewiseDomainIdentificationQA
_register("piecewise_domain_identification", PiecewiseDomainIdentificationQA, "math")
from .pivot_table_qa import PivotTableQA
_register("pivot_table", PivotTableQA, "diagram")
from .planar_judge_qa import PlanarJudgeQA
_register("planar_judge", PlanarJudgeQA, "diagram")
from .plot_function_comparison_qa import PlotFunctionComparisonQA
_register("plot_function_comparison", PlotFunctionComparisonQA, "diagram")
from .polar_function_qa import PolarFunctionQA
_register("polar_function", PolarFunctionQA, "math")
from .polycube_chiral_rotation_qa import PolycubeChiralRotationQA
_register("polycube_chiral_rotation", PolycubeChiralRotationQA, "spatial")
from .polycube_counting_warmup_qa import PolycubeCountingWarmupQA
_register("polycube_counting_warmup", PolycubeCountingWarmupQA, "spatial")
from .polycube_rotation_axis_identify_qa import PolycubeRotationAxisIdentifyQA
_register("polycube_rotation_axis_identify", PolycubeRotationAxisIdentifyQA, "spatial")
from .polycube_rotation_ultra_easy_qa import PolycubeRotationUltraEasyQA
_register("polycube_rotation_ultra_easy", PolycubeRotationUltraEasyQA, "spatial")
from .polygon_area_decompose_qa import PolygonAreaDecomposeQA
_register("polygon_area_decompose", PolygonAreaDecomposeQA, "math")
from .polygon_decomposition_identify_qa import PolygonDecompositionIdentifyQA
_register("polygon_decomposition_identify", PolygonDecompositionIdentifyQA, "math")
from .polygon_interior_angle_advanced_qa import PolygonInteriorAngleAdvancedQA
_register("polygon_interior_angle_advanced", PolygonInteriorAngleAdvancedQA, "math")
from .polygon_rotational_symmetry_qa import PolygonRotationalSymmetryQA
_register("polygon_rotational_symmetry", PolygonRotationalSymmetryQA, "spatial")
from .precise_counting_qa import PreciseCountingQA
_register("precise_counting", PreciseCountingQA, "count")
from .price_tag_qa import PriceTagQA
_register("price_tag", PriceTagQA, "diagram")
from .prim_first_edge_qa import PrimFirstEdgeQA
_register("prim_first_edge", PrimFirstEdgeQA, "diagram")
from .prism_volume_warmup_qa import PrismVolumeWarmupQA
_register("prism_volume_warmup", PrismVolumeWarmupQA, "math")
from .probability_tree_qa import ProbabilityTreeQA
_register("probability_tree", ProbabilityTreeQA, "math")
from .process_flow_diagram_qa import ProcessFlowDiagramQA
_register("process_flow_diagram", ProcessFlowDiagramQA, "diagram")
from .projectile_compare_qa import ProjectileCompareQA
_register("projectile_compare", ProjectileCompareQA, "math")
from .projection_area_comparison_qa import ProjectionAreaComparisonQA
_register("projection_area_comparison", ProjectionAreaComparisonQA, "spatial")
from .projection_view_qa import ProjectionViewQA
_register("projection_view", ProjectionViewQA, "spatial")
from .proportion_ratio_qa import ProportionRatioQA
_register("proportion_ratio", ProportionRatioQA, "math")
from .protractor_read_qa import ProtractorReadQA
_register("protractor_read", ProtractorReadQA, "math")
from .prufer_code_qa import PruferCodeQA
_register("prufer_code", PruferCodeQA, "diagram")
from .ptolemy_quad_qa import PtolemyQuadQA
_register("ptolemy_quad", PtolemyQuadQA, "math")
from .punnett_square_qa import PunnettSquareQA
_register("punnett_square", PunnettSquareQA, "diagram")
from .pythagoras_multistep_qa import PythagorasMultistepQA
_register("pythagoras_multistep", PythagorasMultistepQA, "math")
from .quadrilateral_angle_sum_qa import QuadrilateralAngleSumQA
_register("quadrilateral_angle_sum", QuadrilateralAngleSumQA, "math")
from .quadrilateral_property_chain_qa import QuadrilateralPropertyChainQA
_register("quadrilateral_property_chain", QuadrilateralPropertyChainQA, "math")
from .quantifier_logic_qa import QuantifierLogicQA
_register("quantifier_logic", QuantifierLogicQA, "pattern")
from .rate_of_change_from_graph_qa import RateOfChangeFromGraphQA
_register("rate_of_change_from_graph", RateOfChangeFromGraphQA, "diagram")
from .raven_matrix_qa import RavenMatrixQA
_register("raven_matrix", RavenMatrixQA, "pattern")
from .recolor_ratio_qa import RecolorRatioQA
_register("recolor_ratio", RecolorRatioQA, "math")
from .reflection_path_length_qa import ReflectionPathLengthQA
_register("reflection_path_length", ReflectionPathLengthQA, "spatial")
from .region_count_grid_qa import RegionCountGridQA
_register("region_count_grid", RegionCountGridQA, "count")
from .region_counting_qa import RegionCountingQA
_register("region_counting", RegionCountingQA, "count")
from .relative_direction_chain_qa import RelativeDirectionChainQA
_register("relative_direction_chain", RelativeDirectionChainQA, "spatial")
from .relative_frame_coords_qa import RelativeFrameCoordsQA
_register("relative_frame_coords", RelativeFrameCoordsQA, "spatial")
from .right_triangle_median_qa import RightTriangleMedianQA
_register("right_triangle_median", RightTriangleMedianQA, "math")
from .room_layout_reasoning_qa import RoomLayoutReasoningQA
_register("room_layout_reasoning", RoomLayoutReasoningQA, "spatial")
from .rotate_marked_sheet_qa import RotateMarkedSheetQA
_register("rotate_marked_sheet", RotateMarkedSheetQA, "spatial")
from .rotation_angle_estimation_qa import RotationAngleEstimationQA
_register("rotation_angle_estimation", RotationAngleEstimationQA, "spatial")
from .rotation_chirality_fingerprint_qa import RotationChiralityFingerprintQA
_register("rotation_chirality_fingerprint", RotationChiralityFingerprintQA, "spatial")
from .rotation_composition_mcq_qa import RotationCompositionMcqQA
_register("rotation_composition_mcq", RotationCompositionMcqQA, "spatial")
from .rotation_identification_qa import RotationIdentificationQA
_register("rotation_identification", RotationIdentificationQA, "spatial")
from .rotation_speed_comparison_qa import RotationSpeedComparisonQA
_register("rotation_speed_comparison", RotationSpeedComparisonQA, "spatial")
from .rotation_symmetry_order_3d_qa import RotationSymmetryOrder3dQA
_register("rotation_symmetry_order_3d", RotationSymmetryOrder3dQA, "spatial")
from .rotation_tracking_discrete_qa import RotationTrackingDiscreteQA
_register("rotation_tracking_discrete", RotationTrackingDiscreteQA, "spatial")
from .rule_induction_sequence_qa import RuleInductionSequenceQA
_register("rule_induction_sequence", RuleInductionSequenceQA, "pattern")
from .sankey_diagram_qa import SankeyDiagramQA
_register("sankey_diagram", SankeyDiagramQA, "diagram")
from .scale_balance_qa import ScaleBalanceQA
_register("scale_balance", ScaleBalanceQA, "pattern")
from .scale_drawing_measurement_qa import ScaleDrawingMeasurementQA
_register("scale_drawing_measurement", ScaleDrawingMeasurementQA, "count")
from .scatter_plot_qa import ScatterPlotQA
_register("scatter_plot", ScatterPlotQA, "diagram")
from .scatter_regression_line_qa import ScatterRegressionLineQA
_register("scatter_regression_line", ScatterRegressionLineQA, "diagram")
from .scatter_threshold_count_qa import ScatterThresholdCountQA
_register("scatter_threshold_count", ScatterThresholdCountQA, "diagram")
from .schedule_table_qa import ScheduleTableQA
_register("schedule_table", ScheduleTableQA, "diagram")
from .scientific_graph_interpretation_qa import ScientificGraphInterpretationQA
_register("scientific_graph_interpretation", ScientificGraphInterpretationQA, "diagram")
from .secant_tangent_qa import SecantTangentQA
_register("secant_tangent", SecantTangentQA, "math")
from .semantic_correspondence_qa import SemanticCorrespondenceQA
_register("semantic_correspondence", SemanticCorrespondenceQA, "spatial")
from .sequence_interpolation_qa import SequenceInterpolationQA
_register("sequence_interpolation", SequenceInterpolationQA, "pattern")
from .sequence_next_image_qa import SequenceNextImageQA
_register("sequence_next_image", SequenceNextImageQA, "pattern")
from .set_membership_reasoning_qa import SetMembershipReasoningQA
_register("set_membership_reasoning", SetMembershipReasoningQA, "pattern")
from .set_operation_lists_qa import SetOperationListsQA
_register("set_operation_lists", SetOperationListsQA, "pattern")
from .shadow_from_multiple_lights_qa import ShadowFromMultipleLightsQA
_register("shadow_from_multiple_lights", ShadowFromMultipleLightsQA, "spatial")
from .shadow_overlap_qa import ShadowOverlapQA
_register("shadow_overlap", ShadowOverlapQA, "spatial")
from .shadow_projection_qa import ShadowProjectionQA
_register("shadow_projection", ShadowProjectionQA, "spatial")
from .shadow_similar_triangles_qa import ShadowSimilarTrianglesQA
_register("shadow_similar_triangles", ShadowSimilarTrianglesQA, "spatial")
from .shape_counting_analogy_qa import ShapeCountingAnalogyQA
_register("shape_counting_analogy", ShapeCountingAnalogyQA, "pattern")
from .shape_instance_count_qa import ShapeInstanceCountQA
_register("shape_instance_count", ShapeInstanceCountQA, "count")
from .shape_merge_sequence_qa import ShapeMergeSequenceQA
_register("shape_merge_sequence", ShapeMergeSequenceQA, "pattern")
from .shape_rotation_invariant_qa import ShapeRotationInvariantQA
_register("shape_rotation_invariant", ShapeRotationInvariantQA, "spatial")
from .shape_split_n_pieces_qa import ShapeSplitNPiecesQA
_register("shape_split_n_pieces", ShapeSplitNPiecesQA, "spatial")
from .shape_symmetry_grouping_qa import ShapeSymmetryGroupingQA
_register("shape_symmetry_grouping", ShapeSymmetryGroupingQA, "math")
from .shape_transform_compose_qa import ShapeTransformComposeQA
_register("shape_transform_compose", ShapeTransformComposeQA, "spatial")
from .shingoki_qa import ShingokiQA
_register("shingoki", ShingokiQA, "pattern")
from .shortest_distance_weighted_qa import ShortestDistanceWeightedQA
_register("shortest_distance_weighted", ShortestDistanceWeightedQA, "math")
from .shortest_path_directed_weighted_qa import ShortestPathDirectedWeightedQA
_register("shortest_path_directed_weighted", ShortestPathDirectedWeightedQA, "diagram")
from .shortest_path_visual_qa import ShortestPathVisualQA
_register("shortest_path_visual", ShortestPathVisualQA, "spatial")
from .similar_figure_area_ratio_qa import SimilarFigureAreaRatioQA
_register("similar_figure_area_ratio", SimilarFigureAreaRatioQA, "math")
from .similar_triangles_ratio_qa import SimilarTrianglesRatioQA
_register("similar_triangles_ratio", SimilarTrianglesRatioQA, "math")
from .simpsons_int_qa import SimpsonsIntQA
_register("simpsons_int", SimpsonsIntQA, "math")
from .skyscrapers_solve_qa import SkyscrapersSolveQA
_register("skyscrapers_solve", SkyscrapersSolveQA, "pattern")
from .slice_and_count_qa import SliceAndCountQA
_register("slice_and_count", SliceAndCountQA, "count")
from .sliding_puzzle_qa import SlidingPuzzleQA
_register("sliding_puzzle", SlidingPuzzleQA, "pattern")
from .sliding_sum_blanks_qa import SlidingSumBlanksQA
_register("sliding_sum_blanks", SlidingSumBlanksQA, "math")
from .slope_chart_qa import SlopeChartQA
_register("slope_chart", SlopeChartQA, "diagram")
from .snake_puzzle_qa import SnakePuzzleQA
_register("snake_puzzle", SnakePuzzleQA, "pattern")
from .sokoban_solve_qa import SokobanSolveQA
_register("sokoban_solve", SokobanSolveQA, "pattern")
from .solid_3d_rotation_pick_qa import Solid3DRotationPickQA
_register("solid_3d_rotation_pick", Solid3DRotationPickQA, "spatial")
from .solid_cross_section_identify_qa import SolidCrossSectionIdentifyQA
_register("solid_cross_section_identify", SolidCrossSectionIdentifyQA, "spatial")
from .solid_cross_section_reverse_qa import SolidCrossSectionReverseQA
_register("solid_cross_section_reverse", SolidCrossSectionReverseQA, "spatial")
from .solid_face_edge_vertex_qa import SolidFaceEdgeVertexQA
_register("solid_face_edge_vertex", SolidFaceEdgeVertexQA, "spatial")
from .solid_geometry_qa import SolidGeometryQA
_register("solid_geometry", SolidGeometryQA, "math")
from .sorting_network_trace_qa import SortingNetworkTraceQA
_register("sorting_network_trace", SortingNetworkTraceQA, "diagram")
from .sparkline_table_qa import SparklineTableQA
_register("sparkline_table", SparklineTableQA, "diagram")
from .sparse_image_commonsense_qa import SparseImageCommonsenseQA
_register("sparse_image_commonsense", SparseImageCommonsenseQA, "spatial")
from .spatial_ordering_qa import SpatialOrderingQA
_register("spatial_ordering", SpatialOrderingQA, "spatial")
from .spatial_rotation_qa import SpatialRotationQA
_register("spatial_rotation", SpatialRotationQA, "spatial")
from .sphere_cross_section_area_qa import SphereCrossSectionAreaQA
_register("sphere_cross_section_area", SphereCrossSectionAreaQA, "spatial")
from .spinner_probability_qa import SpinnerProbabilityQA
_register("spinner_probability", SpinnerProbabilityQA, "math")
from .split_image_assembly_qa import SplitImageAssemblyQA
_register("split_image_assembly", SplitImageAssemblyQA, "spatial")
from .stack_queue_trace_qa import StackQueueTraceQA
_register("stack_queue_trace", StackQueueTraceQA, "pattern")
from .stacked_bar_qa import StackedBarQA
_register("stacked_bar", StackedBarQA, "diagram")
from .stacked_cube_touch_count_qa import StackedCubeTouchCountQA
_register("stacked_cube_touch_count", StackedCubeTouchCountQA, "spatial")
from .state_machine_qa import StateMachineQA
_register("state_machine", StateMachineQA, "diagram")
from .stock_buy_sell_max_qa import StockBuySellMaxQA
_register("stock_buy_sell_max", StockBuySellMaxQA, "pattern")
from .stroke_continuity_grouping_qa import StrokeContinuityGroupingQA
_register("stroke_continuity_grouping", StrokeContinuityGroupingQA, "math")
from .structural_pattern_3x3_qa import StructuralPattern3x3QA
_register("structural_pattern_3x3", StructuralPattern3x3QA, "pattern")
from .submersion_water_rise_qa import SubmersionWaterRiseQA
_register("submersion_water_rise", SubmersionWaterRiseQA, "pattern")
from .subplot_condition_count_qa import SubplotConditionCountQA
_register("subplot_condition_count", SubplotConditionCountQA, "diagram")
from .subplot_cross_reasoning_qa import SubplotCrossReasoningQA
_register("subplot_cross_reasoning", SubplotCrossReasoningQA, "diagram")
from .subplot_letter_identify_qa import SubplotLetterIdentifyQA
_register("subplot_letter_identify", SubplotLetterIdentifyQA, "diagram")
from .subplot_position_word_qa import SubplotPositionWordQA
_register("subplot_position_word", SubplotPositionWordQA, "diagram")
from .sudoku_classic_qa import SudokuClassicQA
_register("sudoku_classic", SudokuClassicQA, "pattern")
from .sudoku_visual import SudokuVisualQA
_register("sudoku_visual", SudokuVisualQA, "pattern")
from .supp_comp_angle_qa import SuppCompAngleQA
_register("supp_comp_angle", SuppCompAngleQA, "math")
from .surface_3d_peak_count_qa import Surface3DPeakCountQA
_register("surface_3d_peak_count", Surface3DPeakCountQA, "spatial")
from .syllogism_passage_qa import SyllogismPassageQA
_register("syllogism_passage", SyllogismPassageQA, "pattern")
from .symmetry_detection_qa import SymmetryDetectionQA
_register("symmetry_detection", SymmetryDetectionQA, "pattern")
from .table_cell_lookup_qa import TableCellLookupQA
_register("table_cell_lookup", TableCellLookupQA, "diagram")
from .tangent_line_graph_qa import TangentLineGraphQA
_register("tangent_line_graph", TangentLineGraphQA, "diagram")
from .tapa_qa import TapaQA
_register("tapa", TapaQA, "pattern")
from .temporal_sequence_ordering_qa import TemporalSequenceOrderingQA
_register("temporal_sequence_ordering", TemporalSequenceOrderingQA, "pattern")
from .ternary_plot_qa import TernaryPlotQA
_register("ternary_plot", TernaryPlotQA, "diagram")
from .tessellation_qa import TessellationQA
_register("tessellation", TessellationQA, "math")
from .text_render_math_qa import TextRenderMathQA
_register("text_render_math", TextRenderMathQA, "diagram")
from .three_view_projection_ultra_easy_qa import ThreeViewProjectionUltraEasyQA
_register("three_view_projection_ultra_easy", ThreeViewProjectionUltraEasyQA, "spatial")
from .three_view_projection_warmup_qa import ThreeViewProjectionWarmupQA
_register("three_view_projection_warmup", ThreeViewProjectionWarmupQA, "spatial")
from .threed_coordinate_qa import ThreeDCoordinateQA
_register("threed_coordinate", ThreeDCoordinateQA, "math")
from .timeline_event_ordering_qa import TimelineEventOrderingQA
_register("timeline_event_ordering", TimelineEventOrderingQA, "diagram")
from .topological_sort_qa import TopologicalSortQA
_register("topological_sort", TopologicalSortQA, "diagram")
from .tower_of_hanoi_qa import TowerOfHanoiQA
_register("tower_of_hanoi", TowerOfHanoiQA, "pattern")
from .transformation_chain_prediction_qa import TransformationChainPredictionQA
_register("transformation_chain_prediction", TransformationChainPredictionQA, "math")
from .transformation_composition_qa import TransformationCompositionQA
_register("transformation_composition", TransformationCompositionQA, "math")
from .trapezoid_advanced_qa import TrapezoidAdvancedQA
_register("trapezoid_advanced", TrapezoidAdvancedQA, "math")
from .trapping_rain_water_qa import TrappingRainWaterQA
_register("trapping_rain_water", TrappingRainWaterQA, "pattern")
from .tree_hierarchy_qa import TreeHierarchyQA
_register("tree_hierarchy", TreeHierarchyQA, "diagram")
from .tree_traversal_visit_qa import TreeTraversalVisitQA
_register("tree_traversal_visit", TreeTraversalVisitQA, "diagram")
from .treemap_qa import TreemapQA
_register("treemap", TreemapQA, "diagram")
from .triangle_property_chain_qa import TrianglePropertyChainQA
_register("triangle_property_chain", TrianglePropertyChainQA, "math")
from .triangle_vertex_angle_trace_qa import TriangleVertexAngleTraceQA
_register("triangle_vertex_angle_trace", TriangleVertexAngleTraceQA, "math")
from .trig_word_elevation_qa import TrigWordElevationQA
_register("trig_word_elevation", TrigWordElevationQA, "math")
from .truncated_solid_volume_qa import TruncatedSolidVolumeQA
_register("truncated_solid_volume", TruncatedSolidVolumeQA, "math")
from .truth_table_3variable_qa import TruthTable3variableQA
_register("truth_table_3variable", TruthTable3variableQA, "pattern")
from .truth_table_qa import TruthTableQA
_register("truth_table", TruthTableQA, "pattern")
from .twenty_four_points_qa import TwentyFourPointsQA
_register("twenty_four_points", TwentyFourPointsQA, "pattern")
from .two_lines_intersection_qa import TwoLinesIntersectionQA
_register("two_lines_intersection", TwoLinesIntersectionQA, "math")
from .uk_11plus_chart_qa import UK11PlusChartQA
_register("uk_11plus_chart", UK11PlusChartQA, "diagram")
from .unfold_path_prediction_qa import UnfoldPathPredictionQA
_register("unfold_path_prediction", UnfoldPathPredictionQA, "spatial")
from .unit_conversion_visual_qa import UnitConversionVisualQA
_register("unit_conversion_visual", UnitConversionVisualQA, "math")
from .variable_position_assignment_qa import VariablePositionAssignmentQA
_register("variable_position_assignment", VariablePositionAssignmentQA, "pattern")
from .vector_addition_qa import VectorAdditionQA
_register("vector_addition", VectorAdditionQA, "math")
from .venn_diagram_qa import VennDiagramQA
_register("venn_diagram", VennDiagramQA, "diagram")
from .viewpoint_change_prediction_qa import ViewpointChangePredictionQA
_register("viewpoint_change_prediction", ViewpointChangePredictionQA, "spatial")
from .violin_plot_qa import ViolinPlotQA
_register("violin_plot", ViolinPlotQA, "diagram")
from .vision_only_angle_qa import VisionOnlyAngleQA
_register("vision_only_angle", VisionOnlyAngleQA, "math")
from .vision_only_area_qa import VisionOnlyAreaQA
_register("vision_only_area", VisionOnlyAreaQA, "math")
from .vision_only_coordinate_qa import VisionOnlyCoordinateQA
_register("vision_only_coordinate", VisionOnlyCoordinateQA, "math")
from .vision_only_length_qa import VisionOnlyLengthQA
_register("vision_only_length", VisionOnlyLengthQA, "math")
from .vision_only_triangle_qa import VisionOnlyTriangleQA
_register("vision_only_triangle", VisionOnlyTriangleQA, "math")
from .visual_analogy_abstract_qa import VisualAnalogyAbstractQA
_register("visual_analogy_abstract", VisualAnalogyAbstractQA, "pattern")
from .visual_analogy_raven_3x3_qa import VisualAnalogyRaven3x3QA
_register("visual_analogy_raven_3x3", VisualAnalogyRaven3x3QA, "pattern")
from .visual_counting_ultra_easy_qa import VisualCountingUltraEasyQA
_register("visual_counting_ultra_easy", VisualCountingUltraEasyQA, "count")
from .visual_difference_qa import VisualDifferenceQA
_register("visual_difference", VisualDifferenceQA, "pattern")
from .visual_occlusion_counting_qa import VisualOcclusionCountingQA
_register("visual_occlusion_counting", VisualOcclusionCountingQA, "spatial")
from .visual_penetration_qa import VisualPenetrationQA
_register("visual_penetration", VisualPenetrationQA, "math")
from .visual_rule_exception_qa import VisualRuleExceptionQA
_register("visual_rule_exception", VisualRuleExceptionQA, "pattern")
from .visual_sequence_qa import VisualSequenceQA
_register("visual_sequence", VisualSequenceQA, "pattern")
from .visual_similarity_ranking_qa import VisualSimilarityRankingQA
_register("visual_similarity_ranking", VisualSimilarityRankingQA, "pattern")
from .visual_sudoku_qa import VisualSudokuQA
_register("visual_sudoku", VisualSudokuQA, "pattern")
from .visual_word_problem_warmup_qa import VisualWordProblemWarmupQA
_register("visual_word_problem_warmup", VisualWordProblemWarmupQA, "math")
from .volume_via_displacement_qa import VolumeViaDisplacementQA
_register("volume_via_displacement", VolumeViaDisplacementQA, "math")
from .water_jug_multistep_qa import WaterJugMultistepQA
_register("water_jug_multistep", WaterJugMultistepQA, "pattern")
from .waterfall_chart_qa import WaterfallChartQA
_register("waterfall_chart", WaterfallChartQA, "diagram")
from .word_ladder_qa import WordLadderQA
_register("word_ladder", WordLadderQA, "pattern")
from .wordsearch_qa import WordSearchQA
_register("wordsearch", WordSearchQA, "pattern")
from .yin_yang_grid_qa import YinYangGridQA
_register("yin_yang_grid", YinYangGridQA, "pattern")
