from numpy.random import randint

video_width = 432
video_height = 240

def generateXZ(quadrant,SIZE):
    x = randint(1,SIZE)
    z = randint(1,SIZE)
    if quadrant == 0:
        return x,z
    if quadrant == 1:
        return x,-z
    if quadrant == 2:
        return -x,z
    return -x,-z

def drawTree(x,z):
    return "<DrawBlock x='{}' y='2' z='{}' type='log' />".format(x,z) + \
            "<DrawBlock x='{}' y='3' z='{}' type='log' />".format(x,z) + \
            "<DrawBlock x='{}' y='4' z='{}' type='log' />".format(x,z)

def getXML(MAX_EPISODE_STEPS, SIZE, N_TREES):

    my_xml = ""

    #generate N_TREES * 4 randomly-placed logs per quadrant
    for i in range(4):
        for tree in range(N_TREES):
            x,z = generateXZ(i,SIZE)
            my_xml += drawTree(x,z)

        
        
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <About>
                    <Summary>Survivai Agent</Summary>
                </About>
                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>1000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>''' + \
                            "<DrawCuboid x1='{}' x2='{}' y1='{}' y2='{}' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='{}' y2='1' z1='{}' z2='{}' type='grass'/>".format(-SIZE, SIZE, -SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='4' z1='{}' z2='{}' type='brick_block'/>".format(-SIZE-1, SIZE+1, -SIZE-1, SIZE+1) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='4' z1='{}' z2='{}' type='air'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            my_xml + \
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>
                <AgentSection mode="Survival">
                    <Name>Survivai</Name>
                    <AgentStart>''' + \
                        '<Placement x="{}" y="2" z="{}" pitch="0" yaw="0"/>'.format(0, 0) + \
                        '''
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_axe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <RewardForCollectingItem>
                            <Item reward="100" type="log"/>
                        </RewardForCollectingItem>
                        <RewardForTouchingBlockType>
                            <Block reward="10" type="log"/>
                            <Block reward="-10" type="brick_block"/>
                        </RewardForTouchingBlockType>
                        <ContinuousMovementCommands turnSpeedDegs="60"/>
                        <ObservationFromFullStats/>
                        <DepthProducer>
                            <Width>''' + str(video_width) + '''</Width>
                            <Height>''' + str(video_height) + '''</Height>
                        </DepthProducer>
                        <ColourMapProducer>
                            <Width>''' + str(video_width) + '''</Width>
                            <Height>''' + str(video_height) + '''</Height>
                        </ColourMapProducer>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''