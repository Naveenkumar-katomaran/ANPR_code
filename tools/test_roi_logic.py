import cv2
import numpy as np

def test_roi_logic():
    # Define a simple square ROI
    roi_poly = np.array([
        [100, 100],
        [200, 100],
        [200, 200],
        [100, 200]
    ], dtype=np.int32)

    # Test points
    test_cases = [
        ((150, 150), 1, "Center of ROI"),
        ((100, 100), 0, "Top-left corner"),
        ((150, 100), 0, "On the top edge"),
        ((50, 50), -1, "Outside (top-left)"),
        ((250, 250), -1, "Outside (bottom-right)"),
        ((150, 250), -1, "Outside (below)"),
    ]

    print(f"{'Point':<15} | {'Expected':<10} | {'Result':<10} | {'Status':<10} | {'Description'}")
    print("-" * 80)

    all_passed = True
    for (pt, expected_sign, desc) in test_cases:
        # cv2.pointPolygonTest returns positive for inside, 0 for edge, negative for outside
        result = cv2.pointPolygonTest(roi_poly, (float(pt[0]), float(pt[1])), False)
        
        # Normalize result to sign
        result_sign = 1 if result > 0 else (0 if result == 0 else -1)
        
        status = "PASS" if result_sign == expected_sign else "FAIL"
        if status == "FAIL":
            all_passed = False
            
        print(f"{str(pt):<15} | {expected_sign:<10} | {result_sign:<10} | {status:<10} | {desc}")

    if all_passed:
        print("\n✅ All ROI logic tests passed!")
    else:
        print("\n❌ Some ROI logic tests failed.")

if __name__ == "__main__":
    test_roi_logic()
