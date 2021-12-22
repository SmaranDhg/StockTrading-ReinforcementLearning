# %%
'''---------------------------StockTradingUsingRL---------------------------'''
# goal is to train our DQ-Net 
# creating own environment that simulates the stock market


# first three things:
'''               ___observation___               '''
'''               ___actions___               '''
'''               ___reward___               '''

'''---------------------------Data Representaion---------------------------'''
'''               ___obs___               '''
# n-past bars [opem,high,low,close prices]
# 
# profit or loss we currently have from our current position

# actions
# do nothing
# buy a share
# cloase the position: if noshares-> nothing happens; we pay the commision for the trade


# reward
# 1: reward at buying the share
# 2: reward after closing the position


'''---------------------------Trading Environment---------------------------'''
# 