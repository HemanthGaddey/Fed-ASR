
def client_loop(
        ClientObj,
        ServerIp
):
    
    ClientObj.initialize(ServerIp)
    while True:
        # Training
        ClientObj.get_parameters()
        ClientObj.get_config()
        ClientObj.train()
        ClientObj.send_parameters()

        # Evaluation
        ClientObj.get_parameters()
        ClientObj.get_config()
        ClientObj.eval()
        ClientObj.send_results()