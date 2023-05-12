import cv2
import numpy as np

ix, iy, k = 200, 200, 1


def onMouse(event, x, y, flag, param):
    global ix, iy, k
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        # print("codinates are", x, y)
        k = -1


cv2.namedWindow("window")
cv2.setMouseCallback("window", onMouse)

cap = cv2.VideoCapture(0)
cap.set(3, 740)
cap.set(4, 640)

while True:
    _, frm = cap.read()

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or k == -1 or cv2.getWindowProperty("window", cv2.WND_PROP_VISIBLE) < 1:
        old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        break

old_pts = np.array([[ix, iy]], dtype="float32").reshape(-1, 1, 2)
mask = np.zeros_like(frm)
prev_center = (ix, iy)

while True:
    _, frame2 = cap.read()

    new_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    new_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                    new_gray,
                                                    old_pts,
                                                    None, maxLevel=1,
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                              15, 0.08))

    center = (new_pts.ravel()[0].astype(int), new_pts.ravel()[1].astype(int))
    print(f"The detected points are: {center}")

    # draw a line between previous and current center positions
    cv2.line(mask, prev_center, center, (0, 255, 0), 2)
    prev_center = center

    cv2.circle(mask, center, 2, (0, 255, 0), 2)

    combined = cv2.addWeighted(frame2, 0.7, mask, 0.3, 0.1)

    # cv2.imshow("new win", mask)
    cv2.imshow("wind", combined)

    old_gray = new_gray.copy()
    old_pts = new_pts.copy()

    if cv2.waitKey(1) == 27 or cv2.getWindowProperty("wind", cv2.WND_PROP_VISIBLE) < 1:
        cap.release()
        cv2.destroyAllWindows()
        break
