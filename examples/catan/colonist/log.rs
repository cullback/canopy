//! Parse colonist.io structured game log data into typed events.
//!
//! The game log is extracted from the React virtual scroller as JSON.
//! Each entry has a `text.type` discriminant and type-specific fields.

use crate::game::dev_card::DevCardKind;
use crate::game::resource::{Resource, ResourceArray};

// -- Colonist enum conversions ------------------------------------------------

/// Colonist cardEnum → Resource. Cards 1-5 are resources.
fn card_to_resource(card: u64) -> Option<Resource> {
    match card {
        1 => Some(Resource::Lumber),
        2 => Some(Resource::Brick),
        3 => Some(Resource::Wool),
        4 => Some(Resource::Grain),
        5 => Some(Resource::Ore),
        _ => None,
    }
}

/// Colonist cardEnum → DevCardKind. Cards 10+ are dev cards.
pub(crate) fn card_to_dev(card: u64) -> Option<DevCardKind> {
    match card {
        11 => Some(DevCardKind::Knight),
        12 => Some(DevCardKind::VictoryPoint),
        13 => Some(DevCardKind::Monopoly),
        14 => Some(DevCardKind::RoadBuilding),
        15 => Some(DevCardKind::YearOfPlenty),
        _ => None,
    }
}

/// Convert a JSON array of cardEnums to a ResourceArray.
fn cards_to_resources(cards: &[serde_json::Value]) -> ResourceArray {
    let mut arr = ResourceArray::default();
    for c in cards {
        if let Some(r) = c.as_u64().and_then(card_to_resource) {
            arr[r] += 1;
        }
    }
    arr
}

/// Try to extract a corner coordinate from a log entry's text.
/// Colonist may use field names like "tileCorner", "corner", etc.
fn parse_corner(text: &serde_json::Value) -> Option<CornerCoord> {
    for field in ["tileCorner", "corner", "position"] {
        if let Some(pos) = text.get(field) {
            if let (Some(x), Some(y), Some(z)) = (
                pos.get("x").and_then(|v| v.as_i64()),
                pos.get("y").and_then(|v| v.as_i64()),
                pos.get("z").and_then(|v| v.as_u64()),
            ) {
                return Some((x as i32, y as i32, z as u8));
            }
        }
    }
    // Try direct x, y, z on the text object itself (some log types)
    // But only if pieceEnum indicates a corner piece
    None
}

/// Try to extract an edge coordinate from a log entry's text.
fn parse_edge(text: &serde_json::Value) -> Option<EdgeCoord> {
    for field in ["tileEdge", "edge", "position"] {
        if let Some(pos) = text.get(field) {
            if let (Some(x), Some(y), Some(z)) = (
                pos.get("x").and_then(|v| v.as_i64()),
                pos.get("y").and_then(|v| v.as_i64()),
                pos.get("z").and_then(|v| v.as_u64()),
            ) {
                return Some((x as i32, y as i32, z as u8));
            }
        }
    }
    None
}

/// Player color name for display.
fn color_name(color: u8) -> &'static str {
    match color {
        1 => "Red",
        2 => "Blue",
        3 => "Orange",
        4 => "White",
        5 => "Green",
        _ => "???",
    }
}

// -- Event types --------------------------------------------------------------

/// Corner coordinate (x, y, z) from colonist.io.
pub type CornerCoord = (i32, i32, u8);
/// Edge coordinate (x, y, z) from colonist.io.
pub type EdgeCoord = (i32, i32, u8);

#[derive(Debug)]
pub enum GameEvent {
    // Setup
    PlaceSettlement {
        player: u8,
        corner: Option<CornerCoord>,
    },
    PlaceRoad {
        player: u8,
        edge: Option<EdgeCoord>,
    },
    StartingResources {
        player: u8,
        resources: ResourceArray,
    },

    // Turn structure
    Roll {
        player: u8,
        d1: u8,
        d2: u8,
    },
    RolledSeven,
    BuyDevCard {
        player: u8,
    },

    // Resources
    GotResources {
        player: u8,
        resources: ResourceArray,
    },
    TileBlocked {
        dice_number: u8,
        resource: Option<Resource>,
    },

    // Building
    BuildRoad {
        player: u8,
        edge: Option<EdgeCoord>,
    },
    BuildSettlement {
        player: u8,
        vp: bool,
        corner: Option<CornerCoord>,
    },
    BuildCity {
        player: u8,
        vp: bool,
        corner: Option<CornerCoord>,
    },

    // Robber
    MoveRobber {
        player: u8,
    },
    Stole {
        player: u8,
        victim: u8,
        resources: ResourceArray,
    },
    StoleUnknown {
        player: u8,
        victim: u8,
    },
    StoleNothing {
        player: u8,
    },

    // Dev cards
    PlayedKnight {
        player: u8,
    },
    PlayedMonopoly {
        player: u8,
    },
    PlayedRoadBuilding {
        player: u8,
    },
    PlayedYearOfPlenty {
        player: u8,
    },
    YearOfPlentyGain {
        player: u8,
        resources: ResourceArray,
    },
    PlayedDevCard {
        player: u8,
        card_enum: u64,
    },
    MonopolyResult {
        player: u8,
        count: u8,
        resource: Resource,
    },

    // Discard on 7
    Discard {
        player: u8,
        resources: ResourceArray,
    },

    // Achievements
    LongestRoad {
        player: u8,
    },
    LongestRoadChanged {
        from: u8,
        to: u8,
    },

    // Trading
    TradeOffer {
        player: u8,
        offered: ResourceArray,
        wanted: ResourceArray,
    },
    PlayerTrade {
        player: u8,
        counterparty: u8,
        given: ResourceArray,
        received: ResourceArray,
    },
    BankTrade {
        player: u8,
        given: ResourceArray,
        received: ResourceArray,
    },

    // Embargo
    EmbargoSet {
        player: u8,
        target: u8,
    },
    EmbargoLifted {
        player: u8,
        target: u8,
    },

    // Ignored
    Unknown {
        log_type: u64,
    },
}

// -- Parsing ------------------------------------------------------------------

pub fn parse(entries: &[serde_json::Value]) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let mut placement_dumped = false;
    for entry in entries {
        let text = &entry["text"];
        let Some(log_type) = text["type"].as_u64() else {
            continue;
        };

        // Dump first placement entry to discover coordinate field names
        if !placement_dumped && (log_type == 4 || log_type == 5) {
            eprintln!("--- Raw placement entry (type {log_type}) ---");
            if let Some(obj) = text.as_object() {
                for (k, v) in obj {
                    eprintln!("  text.{k}: {v}");
                }
            }
            placement_dumped = true;
        }

        if let Some(event) = parse_entry(log_type, text, entry) {
            events.push(event);
        }
    }
    events
}

fn parse_entry(
    log_type: u64,
    text: &serde_json::Value,
    entry: &serde_json::Value,
) -> Option<GameEvent> {
    let player = || text["playerColor"].as_u64().unwrap_or(0) as u8;

    match log_type {
        // Separator / system message
        2 | 44 => None,

        // Placement (setup)
        4 => {
            let piece = text["pieceEnum"].as_u64()?;
            let corner = parse_corner(text);
            let edge = parse_edge(text);
            match piece {
                0 => Some(GameEvent::PlaceRoad {
                    player: player(),
                    edge,
                }),
                2 => Some(GameEvent::PlaceSettlement {
                    player: player(),
                    corner,
                }),
                _ => None,
            }
        }

        // Build
        5 => {
            let piece = text["pieceEnum"].as_u64()?;
            let vp = text["isVp"].as_bool().unwrap_or(false);
            let corner = parse_corner(text);
            let edge = parse_edge(text);
            match piece {
                0 => Some(GameEvent::BuildRoad {
                    player: player(),
                    edge,
                }),
                2 => Some(GameEvent::BuildSettlement {
                    player: player(),
                    vp,
                    corner,
                }),
                3 => Some(GameEvent::BuildCity {
                    player: player(),
                    vp,
                    corner,
                }),
                _ => None,
            }
        }

        // Dice roll
        10 => {
            let d1 = text["firstDice"].as_u64()? as u8;
            let d2 = text["secondDice"].as_u64()? as u8;
            Some(GameEvent::Roll {
                player: player(),
                d1,
                d2,
            })
        }

        // Robber moved
        11 => Some(GameEvent::MoveRobber { player: player() }),

        // Stolen from (victim's perspective): from=robber, playerColor=victim
        14 => {
            let victim = player();
            let robber = entry["from"].as_u64()? as u8;
            let cards = text["cardEnums"].as_array()?;
            let resources = cards_to_resources(cards);
            Some(GameEvent::Stole {
                player: robber,
                victim,
                resources,
            })
        }

        // Stole (robber's perspective): from=robber, victim=specificRecipients[0]
        15 => {
            let robber = entry["from"].as_u64()? as u8;
            let victim = entry["specificRecipients"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())? as u8;
            let cards = text["cardEnums"].as_array()?;
            let resources = cards_to_resources(cards);
            Some(GameEvent::Stole {
                player: robber,
                victim,
                resources,
            })
        }

        // Dev card played
        20 => {
            let card_enum = text["cardEnum"].as_u64()?;
            match card_to_dev(card_enum) {
                Some(DevCardKind::Knight) => Some(GameEvent::PlayedKnight { player: player() }),
                Some(DevCardKind::Monopoly) => Some(GameEvent::PlayedMonopoly { player: player() }),
                Some(DevCardKind::RoadBuilding) => {
                    Some(GameEvent::PlayedRoadBuilding { player: player() })
                }
                Some(DevCardKind::YearOfPlenty) => {
                    Some(GameEvent::PlayedYearOfPlenty { player: player() })
                }
                _ => Some(GameEvent::PlayedDevCard {
                    player: player(),
                    card_enum,
                }),
            }
        }

        // Year of Plenty: took resources from bank
        21 => {
            let cards = text["cardEnums"].as_array()?;
            let resources = cards_to_resources(cards);
            Some(GameEvent::YearOfPlentyGain {
                player: player(),
                resources,
            })
        }

        // Resources received
        47 => {
            let cards = text["cardsToBroadcast"].as_array()?;
            let resources = cards_to_resources(cards);
            let dist = text["distributionType"].as_u64().unwrap_or(1);
            if dist == 0 {
                Some(GameEvent::StartingResources {
                    player: player(),
                    resources,
                })
            } else {
                Some(GameEvent::GotResources {
                    player: player(),
                    resources,
                })
            }
        }

        // Tile blocked by robber
        49 => {
            let dn = text["tileInfo"]["diceNumber"].as_u64().unwrap_or(0) as u8;
            let rt = text["tileInfo"]["resourceType"]
                .as_u64()
                .and_then(card_to_resource);
            Some(GameEvent::TileBlocked {
                dice_number: dn,
                resource: rt,
            })
        }

        // Discard on 7
        55 => {
            let cards = text["cardEnums"].as_array()?;
            let resources = cards_to_resources(cards);
            Some(GameEvent::Discard {
                player: player(),
                resources,
            })
        }

        // Seven rolled (no resources)
        60 => Some(GameEvent::RolledSeven),

        // Robber steal: nothing to take
        74 => {
            let p = entry["from"].as_u64().unwrap_or(0) as u8;
            Some(GameEvent::StoleNothing { player: p })
        }

        // Monopoly result
        86 => {
            let count = text["amountStolen"].as_u64()? as u8;
            let resource = text["cardEnum"].as_u64().and_then(card_to_resource)?;
            Some(GameEvent::MonopolyResult {
                player: player(),
                count,
                resource,
            })
        }

        // Bought dev card
        1 => Some(GameEvent::BuyDevCard { player: player() }),

        // Embargo set
        113 => {
            let target = text["embargoedPlayerColor"].as_u64()? as u8;
            Some(GameEvent::EmbargoSet {
                player: player(),
                target,
            })
        }

        // Embargo lifted
        114 => {
            let target = text["embargoedPlayerColor"].as_u64()? as u8;
            Some(GameEvent::EmbargoLifted {
                player: player(),
                target,
            })
        }

        // Player trade
        115 => {
            let counterparty = text["acceptingPlayerColor"].as_u64()? as u8;
            let given = cards_to_resources(text["givenCardEnums"].as_array()?);
            let received = cards_to_resources(text["receivedCardEnums"].as_array()?);
            Some(GameEvent::PlayerTrade {
                player: player(),
                counterparty,
                given,
                received,
            })
        }

        // Bank trade
        116 => {
            let given = cards_to_resources(text["givenCardEnums"].as_array()?);
            let received = cards_to_resources(text["receivedCardEnums"].as_array()?);
            Some(GameEvent::BankTrade {
                player: player(),
                given,
                received,
            })
        }

        // Trade offer
        118 => {
            let offered = cards_to_resources(text["offeredCardEnums"].as_array()?);
            let wanted = cards_to_resources(text["wantedCardEnums"].as_array()?);
            Some(GameEvent::TradeOffer {
                player: player(),
                offered,
                wanted,
            })
        }

        // Longest road awarded
        66 => Some(GameEvent::LongestRoad { player: player() }),

        // Longest road changed hands
        68 => {
            let from = text["playerColorOld"].as_u64()? as u8;
            let to = text["playerColorNew"].as_u64()? as u8;
            Some(GameEvent::LongestRoadChanged { from, to })
        }

        _ => Some(GameEvent::Unknown { log_type }),
    }
}

// -- Display ------------------------------------------------------------------

fn fmt_resources(r: &ResourceArray) -> String {
    let names = ["L", "B", "W", "G", "O"];
    let mut parts = Vec::new();
    for (i, &count) in r.0.iter().enumerate() {
        for _ in 0..count {
            parts.push(names[i]);
        }
    }
    if parts.is_empty() {
        "nothing".to_string()
    } else {
        parts.join(" ")
    }
}

pub fn print(events: &[GameEvent]) {
    let mut hands: std::collections::HashMap<u8, ResourceArray> = std::collections::HashMap::new();
    let mut turn = 0u16;

    for event in events {
        match event {
            GameEvent::Roll { player, d1, d2 } => {
                turn += 1;
                println!(
                    "\n--- Turn {turn}: {} rolls {d1}+{d2}={} ---",
                    color_name(*player),
                    d1 + d2
                );
            }
            GameEvent::RolledSeven => {
                println!("  (seven — no resources)");
            }
            GameEvent::StartingResources { player, resources } => {
                let h = hands.entry(*player).or_default();
                h.add(*resources);
                println!(
                    "  {} starting: {}",
                    color_name(*player),
                    fmt_resources(resources)
                );
            }
            GameEvent::GotResources { player, resources } => {
                let h = hands.entry(*player).or_default();
                h.add(*resources);
                println!("  {} got {}", color_name(*player), fmt_resources(resources));
            }
            GameEvent::PlaceSettlement { player, corner } => {
                let pos = corner
                    .map(|(x, y, z)| format!(" at ({x},{y},{z})"))
                    .unwrap_or_default();
                println!("  {} placed settlement{pos}", color_name(*player));
            }
            GameEvent::PlaceRoad { player, edge } => {
                let pos = edge
                    .map(|(x, y, z)| format!(" at ({x},{y},{z})"))
                    .unwrap_or_default();
                println!("  {} placed road{pos}", color_name(*player));
            }
            GameEvent::BuildRoad { player, edge } => {
                let h = hands.entry(*player).or_default();
                h.sub(crate::game::resource::ROAD_COST);
                let pos = edge
                    .map(|(x, y, z)| format!(" at ({x},{y},{z})"))
                    .unwrap_or_default();
                println!("  {} built road{pos}", color_name(*player));
            }
            GameEvent::BuildSettlement { player, corner, .. } => {
                let h = hands.entry(*player).or_default();
                h.sub(crate::game::resource::SETTLEMENT_COST);
                let pos = corner
                    .map(|(x, y, z)| format!(" at ({x},{y},{z})"))
                    .unwrap_or_default();
                println!("  {} built settlement{pos}", color_name(*player));
            }
            GameEvent::BuildCity { player, corner, .. } => {
                let h = hands.entry(*player).or_default();
                h.sub(crate::game::resource::CITY_COST);
                let pos = corner
                    .map(|(x, y, z)| format!(" at ({x},{y},{z})"))
                    .unwrap_or_default();
                println!("  {} built city{pos}", color_name(*player));
            }
            GameEvent::BuyDevCard { player } => {
                let h = hands.entry(*player).or_default();
                h.sub(crate::game::resource::DEV_CARD_COST);
                println!("  {} bought dev card", color_name(*player));
            }
            GameEvent::MoveRobber { player } => {
                println!("  {} moved robber", color_name(*player));
            }
            GameEvent::Stole {
                player,
                victim,
                resources,
            } => {
                let rh = hands.entry(*player).or_default();
                rh.add(*resources);
                let vh = hands.entry(*victim).or_default();
                vh.sub(*resources);
                println!(
                    "  {} stole {} from {}",
                    color_name(*player),
                    fmt_resources(resources),
                    color_name(*victim)
                );
            }
            GameEvent::StoleUnknown { player, victim } => {
                println!(
                    "  {} stole unknown card from {}",
                    color_name(*player),
                    color_name(*victim)
                );
            }
            GameEvent::StoleNothing { player } => {
                println!("  {} stole nothing", color_name(*player));
            }
            GameEvent::PlayedKnight { player } => {
                println!("  {} played Knight", color_name(*player));
            }
            GameEvent::PlayedMonopoly { player } => {
                println!("  {} played Monopoly", color_name(*player));
            }
            GameEvent::PlayedRoadBuilding { player } => {
                println!("  {} played Road Building", color_name(*player));
            }
            GameEvent::PlayedYearOfPlenty { player } => {
                println!("  {} played Year of Plenty", color_name(*player));
            }
            GameEvent::YearOfPlentyGain { player, resources } => {
                let h = hands.entry(*player).or_default();
                h.add(*resources);
                println!(
                    "  {} took {} from bank",
                    color_name(*player),
                    fmt_resources(resources)
                );
            }
            GameEvent::PlayedDevCard { player, card_enum } => {
                println!(
                    "  {} played dev card (enum {card_enum})",
                    color_name(*player)
                );
            }
            GameEvent::Discard {
                player, resources, ..
            } => {
                let h = hands.entry(*player).or_default();
                h.sub(*resources);
                println!(
                    "  {} discarded {}",
                    color_name(*player),
                    fmt_resources(resources)
                );
            }
            GameEvent::LongestRoad { player } => {
                println!("  {} got Longest Road", color_name(*player));
            }
            GameEvent::LongestRoadChanged { from, to } => {
                println!(
                    "  Longest Road: {} → {}",
                    color_name(*from),
                    color_name(*to)
                );
            }
            GameEvent::MonopolyResult {
                player,
                count,
                resource,
            } => {
                let mut taken = ResourceArray::default();
                taken[*resource] = *count;
                // Remove from all opponents, add to monopolist
                for (&c, h) in hands.iter_mut() {
                    if c != *player {
                        let lost = std::cmp::min(h[*resource], *count);
                        let mut sub = ResourceArray::default();
                        sub[*resource] = lost;
                        h.sub(sub);
                    }
                }
                let h = hands.entry(*player).or_default();
                h.add(taken);
                println!(
                    "  {} monopoly: took {count} {resource}",
                    color_name(*player)
                );
            }
            GameEvent::TradeOffer {
                player,
                offered,
                wanted,
            } => {
                println!(
                    "  {} offers {} for {}",
                    color_name(*player),
                    fmt_resources(offered),
                    fmt_resources(wanted)
                );
            }
            GameEvent::PlayerTrade {
                player,
                counterparty,
                given,
                received,
            } => {
                let ph = hands.entry(*player).or_default();
                ph.sub(*given);
                ph.add(*received);
                let ch = hands.entry(*counterparty).or_default();
                ch.sub(*received);
                ch.add(*given);
                println!(
                    "  {} traded {} for {} with {}",
                    color_name(*player),
                    fmt_resources(given),
                    fmt_resources(received),
                    color_name(*counterparty)
                );
            }
            GameEvent::BankTrade {
                player,
                given,
                received,
            } => {
                let h = hands.entry(*player).or_default();
                h.sub(*given);
                h.add(*received);
                println!(
                    "  {} bank trade: {} → {}",
                    color_name(*player),
                    fmt_resources(given),
                    fmt_resources(received)
                );
            }
            GameEvent::EmbargoSet { player, target } => {
                println!(
                    "  {} embargoed {}",
                    color_name(*player),
                    color_name(*target)
                );
            }
            GameEvent::EmbargoLifted { player, target } => {
                println!(
                    "  {} lifted embargo on {}",
                    color_name(*player),
                    color_name(*target)
                );
            }
            GameEvent::TileBlocked {
                dice_number,
                resource,
            } => {
                let r = resource.map_or("?".to_string(), |r| r.to_string());
                println!("  {r} ({dice_number}) blocked by robber");
            }
            GameEvent::Unknown { log_type } => {
                println!("  [unknown log type {log_type}]");
            }
        }
    }

    // Print final hands
    println!("\n--- Final hands ---");
    let mut players: Vec<_> = hands.iter().collect();
    players.sort_by_key(|(c, _)| *c);
    for (&color, hand) in players {
        println!(
            "  {}: {} (total: {})",
            color_name(color),
            fmt_resources(hand),
            hand.total()
        );
    }
}
