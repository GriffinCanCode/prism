//! Comprehensive tests for the pattern module

use prism_common::span::Position;
use prism_ast::{pattern::*, expr::*, node::*};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};

#[test]
fn test_wildcard_pattern() {
    let wildcard = Pattern::Wildcard;
    
    match wildcard {
        Pattern::Wildcard => {
            // Expected
        }
        _ => panic!("Expected wildcard pattern"),
    }
}

#[test]
fn test_identifier_pattern() {
    let identifier = Pattern::Identifier(Symbol::intern("x"));
    
    match identifier {
        Pattern::Identifier(sym) => {
            assert_eq!(sym, Symbol::intern("x"));
        }
        _ => panic!("Expected identifier pattern"),
    }
}

#[test]
fn test_literal_patterns() {
    // Integer literal pattern
    let int_pattern = Pattern::Literal(LiteralValue::Integer(42));
    
    match int_pattern {
        Pattern::Literal(LiteralValue::Integer(n)) => {
            assert_eq!(n, 42);
        }
        _ => panic!("Expected integer literal pattern"),
    }
    
    // String literal pattern
    let string_pattern = Pattern::Literal(LiteralValue::String("hello".to_string()));
    
    match string_pattern {
        Pattern::Literal(LiteralValue::String(s)) => {
            assert_eq!(s, "hello");
        }
        _ => panic!("Expected string literal pattern"),
    }
    
    // Boolean literal pattern
    let bool_pattern = Pattern::Literal(LiteralValue::Boolean(true));
    
    match bool_pattern {
        Pattern::Literal(LiteralValue::Boolean(b)) => {
            assert!(b);
        }
        _ => panic!("Expected boolean literal pattern"),
    }
    
    // Null literal pattern
    let null_pattern = Pattern::Literal(LiteralValue::Null);
    
    match null_pattern {
        Pattern::Literal(LiteralValue::Null) => {
            // Expected
        }
        _ => panic!("Expected null literal pattern"),
    }
    
    // Money literal pattern
    let money_pattern = Pattern::Literal(LiteralValue::Money {
        amount: 19.99,
        currency: "USD".to_string(),
    });
    
    match money_pattern {
        Pattern::Literal(LiteralValue::Money { amount, currency }) => {
            assert_eq!(amount, 19.99);
            assert_eq!(currency, "USD");
        }
        _ => panic!("Expected money literal pattern"),
    }
    
    // Duration literal pattern
    let duration_pattern = Pattern::Literal(LiteralValue::Duration {
        value: 30.0,
        unit: "seconds".to_string(),
    });
    
    match duration_pattern {
        Pattern::Literal(LiteralValue::Duration { value, unit }) => {
            assert_eq!(value, 30.0);
            assert_eq!(unit, "seconds");
        }
        _ => panic!("Expected duration literal pattern"),
    }
    
    // Regex literal pattern
    let regex_pattern = Pattern::Literal(LiteralValue::Regex(r"\d+".to_string()));
    
    match regex_pattern {
        Pattern::Literal(LiteralValue::Regex(pattern)) => {
            assert_eq!(pattern, r"\d+");
        }
        _ => panic!("Expected regex literal pattern"),
    }
}

#[test]
fn test_tuple_pattern() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let elements = vec![
        AstNode::new(
            Pattern::Identifier(Symbol::intern("x")),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(42)),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Pattern::Wildcard,
            span,
            NodeId::new(3),
        ),
    ];
    
    let tuple_pattern = Pattern::Tuple(elements);
    
    match tuple_pattern {
        Pattern::Tuple(elements) => {
            assert_eq!(elements.len(), 3);
            
            match &elements[0].kind {
                Pattern::Identifier(sym) => {
                    assert_eq!(*sym, Symbol::intern("x"));
                }
                _ => panic!("Expected identifier pattern"),
            }
            
            match &elements[1].kind {
                Pattern::Literal(LiteralValue::Integer(n)) => {
                    assert_eq!(*n, 42);
                }
                _ => panic!("Expected integer literal pattern"),
            }
            
            match &elements[2].kind {
                Pattern::Wildcard => {
                    // Expected
                }
                _ => panic!("Expected wildcard pattern"),
            }
        }
        _ => panic!("Expected tuple pattern"),
    }
}

#[test]
fn test_array_pattern() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let elements = vec![
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(1)),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(2)),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Pattern::Rest(Some(Symbol::intern("rest"))),
            span,
            NodeId::new(3),
        ),
    ];
    
    let array_pattern = Pattern::Array(elements);
    
    match array_pattern {
        Pattern::Array(elements) => {
            assert_eq!(elements.len(), 3);
            
            match &elements[0].kind {
                Pattern::Literal(LiteralValue::Integer(n)) => {
                    assert_eq!(*n, 1);
                }
                _ => panic!("Expected integer literal pattern"),
            }
            
            match &elements[1].kind {
                Pattern::Literal(LiteralValue::Integer(n)) => {
                    assert_eq!(*n, 2);
                }
                _ => panic!("Expected integer literal pattern"),
            }
            
            match &elements[2].kind {
                Pattern::Rest(Some(sym)) => {
                    assert_eq!(*sym, Symbol::intern("rest"));
                }
                _ => panic!("Expected rest pattern with name"),
            }
        }
        _ => panic!("Expected array pattern"),
    }
}

#[test]
fn test_object_pattern() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let field1 = ObjectPatternField {
        key: Symbol::intern("name"),
        pattern: AstNode::new(
            Pattern::Identifier(Symbol::intern("user_name")),
            span,
            NodeId::new(1),
        ),
        optional: false,
    };
    
    let field2 = ObjectPatternField {
        key: Symbol::intern("age"),
        pattern: AstNode::new(
            Pattern::Identifier(Symbol::intern("user_age")),
            span,
            NodeId::new(2),
        ),
        optional: true,
    };
    
    let field3 = ObjectPatternField {
        key: Symbol::intern("email"),
        pattern: AstNode::new(
            Pattern::Wildcard,
            span,
            NodeId::new(3),
        ),
        optional: false,
    };
    
    let object_pattern = Pattern::Object(vec![field1, field2, field3]);
    
    match object_pattern {
        Pattern::Object(fields) => {
            assert_eq!(fields.len(), 3);
            
            // Test first field
            assert_eq!(fields[0].key, Symbol::intern("name"));
            assert!(!fields[0].optional);
            match &fields[0].pattern.kind {
                Pattern::Identifier(sym) => {
                    assert_eq!(*sym, Symbol::intern("user_name"));
                }
                _ => panic!("Expected identifier pattern"),
            }
            
            // Test second field (optional)
            assert_eq!(fields[1].key, Symbol::intern("age"));
            assert!(fields[1].optional);
            match &fields[1].pattern.kind {
                Pattern::Identifier(sym) => {
                    assert_eq!(*sym, Symbol::intern("user_age"));
                }
                _ => panic!("Expected identifier pattern"),
            }
            
            // Test third field
            assert_eq!(fields[2].key, Symbol::intern("email"));
            assert!(!fields[2].optional);
            match &fields[2].pattern.kind {
                Pattern::Wildcard => {
                    // Expected
                }
                _ => panic!("Expected wildcard pattern"),
            }
        }
        _ => panic!("Expected object pattern"),
    }
}

#[test]
fn test_or_pattern() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let patterns = vec![
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(1)),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(2)),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(3)),
            span,
            NodeId::new(3),
        ),
    ];
    
    let or_pattern = Pattern::Or(patterns);
    
    match or_pattern {
        Pattern::Or(patterns) => {
            assert_eq!(patterns.len(), 3);
            
            for (i, pattern) in patterns.iter().enumerate() {
                match &pattern.kind {
                    Pattern::Literal(LiteralValue::Integer(n)) => {
                        assert_eq!(*n, (i + 1) as i64);
                    }
                    _ => panic!("Expected integer literal pattern"),
                }
            }
        }
        _ => panic!("Expected or pattern"),
    }
}

#[test]
fn test_rest_pattern() {
    // Rest pattern with name
    let named_rest = Pattern::Rest(Some(Symbol::intern("remaining")));
    
    match named_rest {
        Pattern::Rest(Some(sym)) => {
            assert_eq!(sym, Symbol::intern("remaining"));
        }
        _ => panic!("Expected named rest pattern"),
    }
    
    // Rest pattern without name
    let unnamed_rest = Pattern::Rest(None);
    
    match unnamed_rest {
        Pattern::Rest(None) => {
            // Expected
        }
        _ => panic!("Expected unnamed rest pattern"),
    }
}

#[test]
fn test_nested_patterns() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create a nested pattern: (x, [1, 2, ...rest])
    let inner_array = vec![
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(1)),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(2)),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Pattern::Rest(Some(Symbol::intern("rest"))),
            span,
            NodeId::new(3),
        ),
    ];
    
    let tuple_elements = vec![
        AstNode::new(
            Pattern::Identifier(Symbol::intern("x")),
            span,
            NodeId::new(4),
        ),
        AstNode::new(
            Pattern::Array(inner_array),
            span,
            NodeId::new(5),
        ),
    ];
    
    let nested_pattern = Pattern::Tuple(tuple_elements);
    
    match nested_pattern {
        Pattern::Tuple(elements) => {
            assert_eq!(elements.len(), 2);
            
            // First element should be identifier
            match &elements[0].kind {
                Pattern::Identifier(sym) => {
                    assert_eq!(*sym, Symbol::intern("x"));
                }
                _ => panic!("Expected identifier pattern"),
            }
            
            // Second element should be array pattern
            match &elements[1].kind {
                Pattern::Array(array_elements) => {
                    assert_eq!(array_elements.len(), 3);
                    
                    // Check array elements
                    match &array_elements[0].kind {
                        Pattern::Literal(LiteralValue::Integer(n)) => {
                            assert_eq!(*n, 1);
                        }
                        _ => panic!("Expected integer literal"),
                    }
                    
                    match &array_elements[1].kind {
                        Pattern::Literal(LiteralValue::Integer(n)) => {
                            assert_eq!(*n, 2);
                        }
                        _ => panic!("Expected integer literal"),
                    }
                    
                    match &array_elements[2].kind {
                        Pattern::Rest(Some(sym)) => {
                            assert_eq!(*sym, Symbol::intern("rest"));
                        }
                        _ => panic!("Expected named rest pattern"),
                    }
                }
                _ => panic!("Expected array pattern"),
            }
        }
        _ => panic!("Expected tuple pattern"),
    }
}

#[test]
fn test_complex_object_pattern() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create a complex object pattern: { name, address: { street, city }, age? }
    let address_fields = vec![
        ObjectPatternField {
            key: Symbol::intern("street"),
            pattern: AstNode::new(
                Pattern::Identifier(Symbol::intern("street_name")),
                span,
                NodeId::new(1),
            ),
            optional: false,
        },
        ObjectPatternField {
            key: Symbol::intern("city"),
            pattern: AstNode::new(
                Pattern::Identifier(Symbol::intern("city_name")),
                span,
                NodeId::new(2),
            ),
            optional: false,
        },
    ];
    
    let fields = vec![
        ObjectPatternField {
            key: Symbol::intern("name"),
            pattern: AstNode::new(
                Pattern::Identifier(Symbol::intern("user_name")),
                span,
                NodeId::new(3),
            ),
            optional: false,
        },
        ObjectPatternField {
            key: Symbol::intern("address"),
            pattern: AstNode::new(
                Pattern::Object(address_fields),
                span,
                NodeId::new(4),
            ),
            optional: false,
        },
        ObjectPatternField {
            key: Symbol::intern("age"),
            pattern: AstNode::new(
                Pattern::Identifier(Symbol::intern("user_age")),
                span,
                NodeId::new(5),
            ),
            optional: true,
        },
    ];
    
    let complex_pattern = Pattern::Object(fields);
    
    match complex_pattern {
        Pattern::Object(fields) => {
            assert_eq!(fields.len(), 3);
            
            // Check name field
            assert_eq!(fields[0].key, Symbol::intern("name"));
            assert!(!fields[0].optional);
            
            // Check address field (nested object)
            assert_eq!(fields[1].key, Symbol::intern("address"));
            assert!(!fields[1].optional);
            match &fields[1].pattern.kind {
                Pattern::Object(address_fields) => {
                    assert_eq!(address_fields.len(), 2);
                    assert_eq!(address_fields[0].key, Symbol::intern("street"));
                    assert_eq!(address_fields[1].key, Symbol::intern("city"));
                }
                _ => panic!("Expected nested object pattern"),
            }
            
            // Check age field (optional)
            assert_eq!(fields[2].key, Symbol::intern("age"));
            assert!(fields[2].optional);
        }
        _ => panic!("Expected object pattern"),
    }
}

#[test]
fn test_pattern_with_or_combinations() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create pattern: (1 | 2 | 3, "success" | "failure")
    let number_patterns = vec![
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(1)),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(2)),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::Integer(3)),
            span,
            NodeId::new(3),
        ),
    ];
    
    let string_patterns = vec![
        AstNode::new(
            Pattern::Literal(LiteralValue::String("success".to_string())),
            span,
            NodeId::new(4),
        ),
        AstNode::new(
            Pattern::Literal(LiteralValue::String("failure".to_string())),
            span,
            NodeId::new(5),
        ),
    ];
    
    let tuple_elements = vec![
        AstNode::new(
            Pattern::Or(number_patterns),
            span,
            NodeId::new(6),
        ),
        AstNode::new(
            Pattern::Or(string_patterns),
            span,
            NodeId::new(7),
        ),
    ];
    
    let complex_or_pattern = Pattern::Tuple(tuple_elements);
    
    match complex_or_pattern {
        Pattern::Tuple(elements) => {
            assert_eq!(elements.len(), 2);
            
            // Check first element (number or pattern)
            match &elements[0].kind {
                Pattern::Or(patterns) => {
                    assert_eq!(patterns.len(), 3);
                    for (i, pattern) in patterns.iter().enumerate() {
                        match &pattern.kind {
                            Pattern::Literal(LiteralValue::Integer(n)) => {
                                assert_eq!(*n, (i + 1) as i64);
                            }
                            _ => panic!("Expected integer literal"),
                        }
                    }
                }
                _ => panic!("Expected or pattern"),
            }
            
            // Check second element (string or pattern)
            match &elements[1].kind {
                Pattern::Or(patterns) => {
                    assert_eq!(patterns.len(), 2);
                    
                    match &patterns[0].kind {
                        Pattern::Literal(LiteralValue::String(s)) => {
                            assert_eq!(s, "success");
                        }
                        _ => panic!("Expected string literal"),
                    }
                    
                    match &patterns[1].kind {
                        Pattern::Literal(LiteralValue::String(s)) => {
                            assert_eq!(s, "failure");
                        }
                        _ => panic!("Expected string literal"),
                    }
                }
                _ => panic!("Expected or pattern"),
            }
        }
        _ => panic!("Expected tuple pattern"),
    }
}

#[test]
fn test_empty_patterns() {
    // Empty tuple pattern
    let empty_tuple = Pattern::Tuple(vec![]);
    
    match empty_tuple {
        Pattern::Tuple(elements) => {
            assert_eq!(elements.len(), 0);
        }
        _ => panic!("Expected empty tuple pattern"),
    }
    
    // Empty array pattern
    let empty_array = Pattern::Array(vec![]);
    
    match empty_array {
        Pattern::Array(elements) => {
            assert_eq!(elements.len(), 0);
        }
        _ => panic!("Expected empty array pattern"),
    }
    
    // Empty object pattern
    let empty_object = Pattern::Object(vec![]);
    
    match empty_object {
        Pattern::Object(fields) => {
            assert_eq!(fields.len(), 0);
        }
        _ => panic!("Expected empty object pattern"),
    }
    
    // Empty or pattern
    let empty_or = Pattern::Or(vec![]);
    
    match empty_or {
        Pattern::Or(patterns) => {
            assert_eq!(patterns.len(), 0);
        }
        _ => panic!("Expected empty or pattern"),
    }
}

#[test]
fn test_pattern_cloning() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let original_pattern = AstNode::new(
        Pattern::Identifier(Symbol::intern("test")),
        span,
        NodeId::new(1),
    );
    
    let cloned_pattern = original_pattern.clone();
    
    assert_eq!(original_pattern.span, cloned_pattern.span);
    assert_eq!(original_pattern.id, cloned_pattern.id);
    
    match (&original_pattern.kind, &cloned_pattern.kind) {
        (Pattern::Identifier(sym1), Pattern::Identifier(sym2)) => {
            assert_eq!(sym1, sym2);
        }
        _ => panic!("Pattern cloning failed"),
    }
}

#[test]
fn test_object_pattern_field_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let field = ObjectPatternField {
        key: Symbol::intern("test_field"),
        pattern: AstNode::new(
            Pattern::Wildcard,
            span,
            NodeId::new(1),
        ),
        optional: true,
    };
    
    assert_eq!(field.key, Symbol::intern("test_field"));
    assert!(field.optional);
    
    match &field.pattern.kind {
        Pattern::Wildcard => {
            // Expected
        }
        _ => panic!("Expected wildcard pattern"),
    }
}

#[test]
fn test_pattern_exhaustiveness_examples() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Example patterns that would be used in match expressions
    
    // Pattern for matching HTTP status codes
    let http_status_patterns = vec![
        AstNode::new(
            Pattern::Or(vec![
                AstNode::new(
                    Pattern::Literal(LiteralValue::Integer(200)),
                    span,
                    NodeId::new(1),
                ),
                AstNode::new(
                    Pattern::Literal(LiteralValue::Integer(201)),
                    span,
                    NodeId::new(2),
                ),
                AstNode::new(
                    Pattern::Literal(LiteralValue::Integer(204)),
                    span,
                    NodeId::new(3),
                ),
            ]),
            span,
            NodeId::new(4),
        ),
        AstNode::new(
            Pattern::Or(vec![
                AstNode::new(
                    Pattern::Literal(LiteralValue::Integer(400)),
                    span,
                    NodeId::new(5),
                ),
                AstNode::new(
                    Pattern::Literal(LiteralValue::Integer(404)),
                    span,
                    NodeId::new(6),
                ),
            ]),
            span,
            NodeId::new(7),
        ),
        AstNode::new(
            Pattern::Or(vec![
                AstNode::new(
                    Pattern::Literal(LiteralValue::Integer(500)),
                    span,
                    NodeId::new(8),
                ),
                AstNode::new(
                    Pattern::Literal(LiteralValue::Integer(503)),
                    span,
                    NodeId::new(9),
                ),
            ]),
            span,
            NodeId::new(10),
        ),
        AstNode::new(
            Pattern::Wildcard, // Catch-all for other status codes
            span,
            NodeId::new(11),
        ),
    ];
    
    assert_eq!(http_status_patterns.len(), 4);
    
    // Verify the structure
    match &http_status_patterns[0].kind {
        Pattern::Or(success_codes) => {
            assert_eq!(success_codes.len(), 3);
        }
        _ => panic!("Expected or pattern for success codes"),
    }
    
    match &http_status_patterns[3].kind {
        Pattern::Wildcard => {
            // Expected catch-all pattern
        }
        _ => panic!("Expected wildcard catch-all pattern"),
    }
} 