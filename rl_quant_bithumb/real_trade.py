from rl_quant_bithumb.bithumb_broker import *

def main(conkey, seckey, target_currency, log_interval):
    broker = RealBroker(conkey, seckey, target_currency, log_interval)
    broker.trade()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--conkey", type=str)
    parser.add_argument("--seckey", type=str)
    parser.add_argument("--target_currency", type=str)
    parser.add_argument("--log_interval", type=float)

    args = parser.parse_args()
    
    main(args.conkey, args.seckey, args.target_currency, args.log_interval)
