from sentinelhub import SHConfig

config = SHConfig()
config.sh_client_id = 'your_sh_client_id'
config.sh_client_secret = 'your_sh_client_secret'
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
config.save("my-profile")
