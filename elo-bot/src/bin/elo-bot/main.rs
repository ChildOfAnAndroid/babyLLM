#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SettingsApi {
	twitch_api: String,
	twitch_client_id: String,
	twitch_client_secret: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SettingsTestMode {
	test_mode: bool,
    user_id: i32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
	debug: bool,
	apis: SettingsApi,
    test: SettingsTestMode,
}

impl Default for Settings {
	fn default() -> Self {
		Self {
			debug: false,
            apis: SettingsApi {
                twitch_api: "".to_string(),
                twitch_client_id: "".to_string(),
                twitch_client_secret: "".to_string(),
            },
            test: SettingsTestMode {
                test_mode: false,
                user_id: 0,
            },
		}
	}
}
impl Settings {
	fn twitch_api(&self) -> String {
		self.apis.twitch_api.clone()
	}
	fn twitch_client_id(&self) -> String {
		self.apis.twitch_client_id.clone()
	}
	fn twitch_client_secret(&self) -> String {
		self.apis.twitch_client_secret.clone()
	}
    fn test_twitch_user_id(&self) -> String {
        self.test.user_id.to_string()
    }
}

#[tokio::main]
async fn main() -> Result<(), eyre::Report> {
    run().await;

    Ok(())
}

pub async fn run() -> eyre::Result<()> {
    println!("Hello, world!");

	let settings = config::Config::builder()
			.add_source(config::File::with_name("config").required(false))
			.add_source(config::Environment::with_prefix("ELOBOT"))
			.build()
			.unwrap();

	let settings: Settings = settings.try_deserialize().expect("Deserialization failed");
	println!("{}", settings.twitch_api());
	println!("{settings:#?}");

	let client: twitch_api::TwitchClient<reqwest::Client> = twitch_api::TwitchClient::default();
    let token = twitch_api::twitch_oauth2::AppAccessToken::get_app_access_token(
        &client,
        settings.twitch_client_id().into(),
        settings.twitch_client_secret().into(),
        vec![/* scopes */],
    )
    .await?;
    let req = twitch_api::helix::channels::GetChannelInformationRequest::broadcaster_ids(&[settings.test_twitch_user_id()]);
    println!(
        "{:?}",
        &client.helix.req_get(req, &token).await?.data[0].title
    );

    Ok(())
}