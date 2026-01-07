bool HasFinished() {
    auto app = cast<CGameManiaPlanet>(GetApp());
    auto pg  = cast<CGameCtnPlayground>(app.CurrentPlayground);
    if (pg is null || pg.GameTerminals.Length == 0) return false;

    auto seq = pg.GameTerminals[0].UISequence_Current;

    if (seq == CGamePlaygroundUIConfig::EUISequence::Finish)
        return true;


    if (seq == CGamePlaygroundUIConfig::EUISequence::EndRound
     || seq == CGamePlaygroundUIConfig::EUISequence::Podium
     || seq == CGamePlaygroundUIConfig::EUISequence::Outro)
        return true;

    return false;
}

void Main() {
    uint16 serverPort = 12345;
    auto serverSocket = Net::Socket();
    
    if (!serverSocket.Listen(serverPort)) {
        print("Serwer nie działa :(");
        return;
    }
    
    print("Serwer działa :)");
    print("Oczekiwanie na klienta.");

    while (true) {
        auto clientSocket = serverSocket.Accept();
        if (clientSocket is null) {
            yield();
            continue;
        }

        print("Połączono.");
		MemoryBuffer@ buf = MemoryBuffer(0);

        while (clientSocket.IsReady()) {
            yield();
            if(clientSocket.IsHungUp())
            {
                break;
            }
            auto app = cast<CGameManiaPlanet>(GetApp());
            if (app !is null && app.GameScene !is null) {
                auto sceneVis = app.GameScene;
                auto player = VehicleState::GetViewingPlayer();
                if (player is null) {
                    continue;
                }

                auto vehicleVis = VehicleState::GetVis(sceneVis, player);
                if (vehicleVis !is null) {
                    
					buf.Seek(0, 0);
                    //buf.Write(vehicleVis.AsyncState.InputSteer);
                    //buf.Write(vehicleVis.AsyncState.InputGasPedal);
                    //buf.Write(vehicleVis.AsyncState.InputBrakePedal);
                    //buf.Write(vehicleVis.AsyncState.InputIsBraking ? 1 : 0);
                    //buf.Write(vehicleVis.AsyncState.IsTurbo ? 1 : 0);
                    //buf.Write(vehicleVis.AsyncState.TurboTime);
                    //buf.Write(vehicleVis.AsyncState.ReactorBoostLvl);
                    //buf.Write(vehicleVis.AsyncState.ReactorBoostType);
                    
                    buf.Write(vehicleVis.AsyncState.Position.x);
                    buf.Write(vehicleVis.AsyncState.Position.y);
                    buf.Write(vehicleVis.AsyncState.Position.z);

                    //buf.Write(vehicleVis.AsyncState.Dir.x);
                    //buf.Write(vehicleVis.AsyncState.Dir.y);
                    //buf.Write(vehicleVis.AsyncState.Dir.z);

                    //buf.Write(vehicleVis.AsyncState.Up.x);
                    //buf.Write(vehicleVis.AsyncState.Up.y);
                    //buf.Write(vehicleVis.AsyncState.Up.z);

                    //buf.Write(vehicleVis.AsyncState.WorldVel.x);
                    //buf.Write(vehicleVis.AsyncState.WorldVel.y);
                    //buf.Write(vehicleVis.AsyncState.WorldVel.z);

                    buf.Write(vehicleVis.AsyncState.FrontSpeed * 3.6);
                    //buf.Write(vehicleVis.AsyncState.IsGroundContact ? 1 : 0);
                    //buf.Write(vehicleVis.AsyncState.GroundDist);
                    //buf.Write(vehicleVis.AsyncState.FLGroundContactMaterial);
                    //buf.Write(vehicleVis.AsyncState.WetnessValue01);
                    //buf.Write(vehicleVis.AsyncState.WaterImmersionCoef);
                    buf.Write(HasFinished() ? 1 : 0);
					buf.Seek(0, 0);
                    if (!clientSocket.Write(buf)) {
                        print("Nie mozna wyslac danych.");
                        clientSocket.Close();
                    }
                } else {
                    continue;
                }
            }
        }

        clientSocket.Close();
        print("Połączenie zakończone.");
        print("Oczekiwanie na nowego klienta.");
    }

    serverSocket.Close();
    print("Serwer zamknięty.");
}
