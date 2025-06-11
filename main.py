# main function

from stage_1 import stage_1_main
from stage_2 import stage_2_main
from stage_3 import stage_3_main
from draw_result import draw_result_main

def main():
    """
    steps:
        1. 讀取frame
        2. Stage_1模型(定位排裝位置),之後更新為定位排裝、針劑、餐包位置
            --> input(image)：frame
            --> output(image)：crop box/boxes from model(frame)
        3. Stage_2模型
            --> input(image)：box
            --> output(dict)：{'box': front or back}
        4. Stage_3模型
            --> input(image, dict)：output from Stage_1, output from Stage_2
            --> output(dict)：result of box/boxes
        5. draw result
            --> input(image): box/boxes, output from Stage_3
            --> output(image): result drawing on box
        6. plt.show(draw result)
    """