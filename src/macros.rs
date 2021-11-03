#[macro_export]
macro_rules! accessor {
    ((get = $property:ident): $type:ty) => {
        fn $property(&self) -> $type;
    };
    ((get = $property:ident, set = $setter:ident): $type:ty) => {
        fn $property(&self) -> $type;
        fn $setter(&mut self, val: $type);
    };
}

#[macro_export]
macro_rules! accessor_impl {
    ((get = $property:ident): $type: ty) => {
        fn $property(&self) -> $type {
            self.$property
        }
    };
    ((get = $property:ident, set = $setter:ident): $type: ty) => {
        fn $property(&self) -> $type {
            self.$property
        }
        fn $setter(&mut self, val: $type) {
            self.$property = val;
        }
    };
}
